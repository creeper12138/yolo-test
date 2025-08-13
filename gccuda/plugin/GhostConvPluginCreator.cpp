#include "GhostConvPlugin.h"
#include "NvInfer.h"
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace ghostconv;

namespace
{
	const char* GHOST_CONV_PLUGIN_NAME = "GhostConv_TRT";
	const char* GHOST_CONV_PLUGIN_VERSION = "1";
} // namespace

class GhostConvPluginCreator : public IPluginCreator
{
public:
	GhostConvPluginCreator()
	{
		// 可以在这里初始化属性字段描述
		mPluginAttributes.clear();
	}

	const char* getPluginName() const noexcept override
	{
		return GHOST_CONV_PLUGIN_NAME;
	}

	const char* getPluginVersion() const noexcept override
	{
		return GHOST_CONV_PLUGIN_VERSION;
	}

	const PluginFieldCollection* getFieldNames() noexcept override
	{
		mFC.nbFields = mPluginAttributes.size();
		mFC.fields = mPluginAttributes.data();
		return &mFC;
	}

	IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
	{
		try
		{
			return new GhostConvPlugin();
		}
		catch (...)
		{
			return nullptr;
		}
	}

	IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
	{
		try
		{
			return new GhostConvPlugin(serialData, serialLength);
		}
		catch (...)
		{
			return nullptr;
		}
	}

	void setPluginNamespace(const char* pluginNamespace) noexcept override
	{
		mNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const noexcept override
	{
		return mNamespace.c_str();
	}

private:
	std::string mNamespace;
	std::vector<PluginField> mPluginAttributes;
	PluginFieldCollection mFC{};
};

// 通过这个宏注册插件，TensorRT 会自动识别
REGISTER_TENSORRT_PLUGIN(GhostConvPluginCreator);
