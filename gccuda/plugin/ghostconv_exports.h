#pragma once

// ── 选择是否导出符号 ─────────────────────────────────────────────
// 在构建本共享库时定义 GC_BUILD_SHARED
// 如果是静态库或不需要导出，定义 GC_STATIC（会清空导出装饰）

#if defined(GC_STATIC)
	#define GC_EXPORT
	#define GC_LOCAL
#else
	#if defined(_WIN32) || defined(_WIN64)
		#if defined(GC_BUILD_SHARED)
			#define GC_EXPORT __declspec(dllexport)
		#else
			#define GC_EXPORT __declspec(dllimport)
		#endif
		#define GC_LOCAL
	#else
		#if __GNUC__ >= 4
			#define GC_EXPORT __attribute__((visibility("default")))
			#define GC_LOCAL  __attribute__((visibility("hidden")))
		#else
			#define GC_EXPORT
			#define GC_LOCAL
		#endif
	#endif
#endif

// ── C / C++ 兼容 ─────────────────────────────────────────────────
#ifdef __cplusplus
	#define GC_EXTERN_C extern "C"
#else
	#define GC_EXTERN_C
#endif

// ── 最终暴露给外部使用的宏 ───────────────────────────────────────
// C 接口导出（函数）: extern "C" + 导出
#define GC_API       GC_EXTERN_C GC_EXPORT
// C++ 符号导出（类/变量/函数）
#define GC_CLASS_API GC_EXPORT
