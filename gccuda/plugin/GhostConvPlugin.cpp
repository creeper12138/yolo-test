#include "GhostConvPlugin.h"
#include <cstring>
#include <cassert>
#include <iostream>

namespace ghostconv {

// ─────────── 构造函数 ───────────
GhostConvPlugin::GhostConvPlugin(const GhostConvParams& params,
	const std::vector<float>& w1,
	const std::vector<float>& w2)
	: mParams(params), mW1Host(w1), mW2Host(w2) {
	mW1Bytes = mW1Host.size() * sizeof(float);
	mW2Bytes = mW2Host.size() * sizeof(float);
}

GhostConvPlugin::GhostConvPlugin(const void* data, size_t length) {
	const char* p = reinterpret_cast<const char*>(data);
	read(p, mParams.k);
	read(p, mParams.s);
	read(p, mParams.p);
	read(p, mParams.cMid);
	read(p, mParams.dtype);
	read(p, mW1Bytes);
	read(p, mW2Bytes);
	mW1Host.resize(mW1Bytes / sizeof(float));
	mW2Host.resize(mW2Bytes / sizeof(float));
	std::memcpy(mW1Host.data(), p, mW1Bytes); p += mW1Bytes;
	std::memcpy(mW2Host.data(), p, mW2Bytes); p += mW2Bytes;
}

// ─────────── IPluginV2 基础接口 ───────────
const char* GhostConvPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }
const char* GhostConvPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
int GhostConvPlugin::getNbOutputs() const noexcept { return 1; }

int GhostConvPlugin::initialize() noexcept {
	if (mW1Bytes) {
		cudaMalloc(&mW1Dev, mW1Bytes);
		cudaMemcpy(mW1Dev, mW1Host.data(), mW1Bytes, cudaMemcpyHostToDevice);
	}
	if (mW2Bytes) {
		cudaMalloc(&mW2Dev, mW2Bytes);
		cudaMemcpy(mW2Dev, mW2Host.data(), mW2Bytes, cudaMemcpyHostToDevice);
	}
	return 0;
}

void GhostConvPlugin::terminate() noexcept {
	if (mW1Dev) { cudaFree(mW1Dev); mW1Dev = nullptr; }
	if (mW2Dev) { cudaFree(mW2Dev); mW2Dev = nullptr; }
}

size_t GhostConvPlugin::getSerializationSize() const noexcept {
	return sizeof(mParams.k) + sizeof(mParams.s) + sizeof(mParams.p) + sizeof(mParams.cMid) +
	       sizeof(mParams.dtype) +
	       sizeof(mW1Bytes) + sizeof(mW2Bytes) + mW1Bytes + mW2Bytes;
}

void GhostConvPlugin::serialize(void* buffer) const noexcept {
	char* p = reinterpret_cast<char*>(buffer);
	write(p, mParams.k);
	write(p, mParams.s);
	write(p, mParams.p);
	write(p, mParams.cMid);
	write(p, mParams.dtype);
	write(p, mW1Bytes);
	write(p, mW2Bytes);
	std::memcpy(p, mW1Host.data(), mW1Bytes); p += mW1Bytes;
	std::memcpy(p, mW2Host.data(), mW2Bytes); p += mW2Bytes;
}

nvinfer1::IPluginV2DynamicExt* GhostConvPlugin::clone() const noexcept {
	auto* plugin = new GhostConvPlugin(mParams, mW1Host, mW2Host);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

void GhostConvPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace ? pluginNamespace : ""; }
const char* GhostConvPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// ─────────── IPluginV2Ext ───────────
nvinfer1::DataType GhostConvPlugin::getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept {
	return inputTypes[0];
}

// ─────────── IPluginV2DynamicExt ───────────
nvinfer1::DimsExprs GhostConvPlugin::getOutputDimensions(int32_t, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
	assert(nbInputs == 1);
	auto in = inputs[0];
	nvinfer1::DimsExprs out;
	out.nbDims = 4;
	out.d[0] = in.d[0]; // N
	out.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
		*exprBuilder.constant(2), *exprBuilder.constant(mParams.cMid)); // C_out = 2*C_mid

	auto H2 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *in.d[2], *exprBuilder.constant(2 * mParams.p));
	auto H3 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *H2, *exprBuilder.constant(mParams.k));
	auto H4 = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *H3, *exprBuilder.constant(mParams.s));
	out.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *H4, *exprBuilder.constant(1));

	auto W2 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *in.d[3], *exprBuilder.constant(2 * mParams.p));
	auto W3 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *W2, *exprBuilder.constant(mParams.k));
	auto W4 = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *W3, *exprBuilder.constant(mParams.s));
	out.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *W4, *exprBuilder.constant(1));
	return out;
}

bool GhostConvPlugin::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t, int32_t) noexcept {
	const auto& desc = inOut[pos];
	if (pos == 0) {
		return (desc.format == nvinfer1::TensorFormat::kLINEAR) &&
		       (desc.type == nvinfer1::DataType::kFLOAT); // 初期只支持 FP32
	}
	if (pos == 1) {
		const auto& in0 = inOut[0];
		return (desc.format == in0.format) && (desc.type == in0.type);
	}
	return false;
}

void GhostConvPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t, const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept {
	mComputeType = in[0].desc.type;
}

size_t GhostConvPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int32_t, const nvinfer1::PluginTensorDesc*, int32_t) const noexcept {
	return 0; // 我们内核不需要额外 workspace
}

int GhostConvPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
	const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs,
	void* const* outputs,
	void* workspace,
	cudaStream_t stream) noexcept {

	int N = inputDesc[0].dims.d[0];
	int C_in = inputDesc[0].dims.d[1];
	int H = inputDesc[0].dims.d[2];
	int W = inputDesc[0].dims.d[3];

	int H_out, W_out;
	ghostconv_get_output_shape(H, W, mParams.k, mParams.s, mParams.p, &H_out, &W_out);

	bool ok = ghostconv_launch(
		inputs[0], outputs[0],
		mW1Dev, mW2Dev,
		N, C_in, H, W,
		mParams.cMid,
		mParams.k, mParams.s, mParams.p,
		H_out, W_out,
		mParams.dtype,
		workspace, 0,
		stream
	);
	return ok ? 0 : -1;
}

void GhostConvPlugin::destroy() noexcept { delete this; }

// ─────────── Creator ───────────
GhostConvPluginCreator::GhostConvPluginCreator() {
	mFields.reserve(6);
	mFields.emplace_back(nvinfer1::PluginField{"k", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
	mFields.emplace_back(nvinfer1::PluginField{"s", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
	mFields.emplace_back(nvinfer1::PluginField{"p", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
	mFields.emplace_back(nvinfer1::PluginField{"cMid", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
	mFields.emplace_back(nvinfer1::PluginField{"w1", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
	mFields.emplace_back(nvinfer1::PluginField{"w2", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
	mFC.nbFields = static_cast<int>(mFields.size());
	mFC.fields = mFields.data();
}

const char* GhostConvPluginCreator::getPluginName() const noexcept { return kPLUGIN_NAME; }
const char* GhostConvPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
const nvinfer1::PluginFieldCollection* GhostConvPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2* GhostConvPluginCreator::createPlugin(const char*, const nvinfer1::PluginFieldCollection* fc) noexcept {
	GhostConvParams params{};
	std::vector<float> w1, w2;
	for (int i = 0; i < fc->nbFields; ++i) {
		const auto& f = fc->fields[i];
		if (!strcmp(f.name, "k")) params.k = *static_cast<const int*>(f.data);
		else if (!strcmp(f.name, "s")) params.s = *static_cast<const int*>(f.data);
		else if (!strcmp(f.name, "p")) params.p = *static_cast<const int*>(f.data);
		else if (!strcmp(f.name, "cMid")) params.cMid = *static_cast<const int*>(f.data);
		else if (!strcmp(f.name, "w1")) w1.assign(static_cast<const float*>(f.data), static_cast<const float*>(f.data) + f.length);
		else if (!strcmp(f.name, "w2")) w2.assign(static_cast<const float*>(f.data), static_cast<const float*>(f.data) + f.length);
	}
	params.dtype = GC_DTYPE_F32; // 初期固定 FP32
	auto* plugin = new GhostConvPlugin(params, w1, w2);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

nvinfer1::IPluginV2* GhostConvPluginCreator::deserializePlugin(const char*, const void* serialData, size_t serialLength) noexcept {
	auto* plugin = new GhostConvPlugin(serialData, serialLength);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

void GhostConvPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace ? libNamespace : ""; }
const char* GhostConvPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// 注册插件
REGISTER_TENSORRT_PLUGIN(GhostConvPluginCreator);

} // namespace ghostconv
