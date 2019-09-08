# Description:
#   TensorFlow C++ inference example for labeling images.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

exports_files(["data/grace_hopper.jpg"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "gm_label_image",
    srcs = [
        "main.cpp",
        "GraphManager.cpp",
    ],
    includes = [
        "GraphManager.hpp",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "ARFramework_tools_test",
    srcs = [
        "test.cpp",
        "grid_tools.cpp",
    ],
    includes = [
        "grid_tools.hpp",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

