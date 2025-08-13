#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include "ghostconv_api.h"        // 我们刚做的统一 C API
#include "ghostconv_exports.h"    // 导出宏

namespace ghostconv {

static constexpr const char* kPLUGIN_NAME    = "GhostConv_TRT";
static constexpr const char* kPLUGIN_VERSION = "1";

struct GhostConvParams {
	int k{3};
	int s{1};
	int p{1};
	int cMid{0};
	gc_dtype_t dtype{GC_DTYPE_F32}; // 初期只用 FP32
};

class GC_CLASS_API GhostConvPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
	// API 构图直接构造
	GhostConvPlugin(const GhostConvParams& params,
	                const std::vector<float>& w1,
	                const std::vector<float>& w2);

	// 反序列化构造
	GhostConvPlugin(const void* data, size_t length);

	// 禁拷贝
	GhostConvPlugin(const GhostConvPlugin&) = delete;
	GhostConvPlugin& operator=(const GhostConvPlugin&) = delete;

	// ---- IPluginV2 ----
	const char* getPluginType() const noexcept override;
	const char* getPluginVersion() const noexcept override;
	int getNbOutputs() const noexcept override;
	int initialize() noexcept override;
	void terminate() noexcept override;
	size_t getSerializationSize() const noexcept override;
	void serialize(void* buffer) const noexcept override;
	void destroy() noexcept override;
	nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
	void setPluginNamespace(const char* pluginNamespace) noexcept override;
	const char* getPluginNamespace() const noexcept override;

	// ---- IPluginV2Ext ----
	nvinfer1::DataType getOutputDataType(int index,
		const nvinfer1::DataType* inputTypes,
		int nbInputs) const noexcept override;

	// ---- IPluginV2DynamicExt ----
	nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
		const nvinfer1::DimsExprs* inputs,
		int32_t nbInputs,
		nvinfer1::IExprBuilder& exprBuilder) noexcept override;

	bool supportsFormatCombination(int32_t pos,
		const nvinfer1::PluginTensorDesc* inOut,
		int32_t nbInputs,
		int32_t nbOutputs) noexcept override;

	void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
		int32_t nbInputs,
		const nvinfer1::DynamicPluginTensorDesc* out,
		int32_t nbOutputs) noexcept override;

	size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
		int32_t nbInputs,
		const nvinfer1::PluginTensorDesc* outputs,
		int32_t nbOutputs) const noexcept override;

	int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
		const nvinfer1::PluginTensorDesc* outputDesc,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream) noexcept override;

private:
	template <typename T>
	static void write(char*& p, const T& val) {
		*reinterpret_cast<T*>(p) = val;
		p += sizeof(T);
	}
	template <typename T>
	static void read(const char*& p, T& val) {
		val = *reinterpret_cast<const T*>(p);
		p += sizeof(T);
	}

private:
	GhostConvParams mParams{};
	std::vector<float> mW1Host, mW2Host;
	void* mW1Dev{nullptr};
	void* mW2Dev{nullptr};
	size_t mW1Bytes{0};
	size_t mW2Bytes{0};
	std::string mNamespace;
	nvinfer1::DataType mComputeType{nvinfer1::DataType::kFLOAT};
};

// 插件 Creator
class GC_CLASS_API GhostConvPluginCreator : public nvinfer1::IPluginCreator {
public:
	GhostConvPluginCreator();
	const char* getPluginName() const noexcept override;
	const char* getPluginVersion() const noexcept override;
	const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
	nvinfer1::IPluginV2* createPlugin(const char* name,
		const nvinfer1::PluginFieldCollection* fc) noexcept override;
	nvinfer1::IPluginV2* deserializePlugin(const char* name,
		const void* serialData,
		size_t serialLength) noexcept override;
	void setPluginNamespace(const char* libNamespace) noexcept override;
	const char* getPluginNamespace() const noexcept override;

private:
	std::string mNamespace;
	nvinfer1::PluginFieldCollection mFC{};
	std::vector<nvinfer1::PluginField> mFields;
};

} // namespace ghostconv
