#!/usr/bin/env python3
"""
Generates benchmark and sample ONNX models for Bend ONNX runtime testing.
"""

import sys
for p in [
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/site-packages",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import numpy as np
import onnx
from onnx import helper, TensorProto

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_linear_model():
    """Simple linear regression: Y = X * W + B (1x4 -> 1x2)"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
    
    # W is 4x2 in standard matmul or 2x4 with transB=1
    np.random.seed(42)
    w_data = np.array([
        [0.2, -0.5],
        [0.4, 0.1],
        [-0.3, 0.7],
        [0.8, -0.2]
    ], dtype=np.float32)
    # Stored as [2, 4] for Gemm with transB=1 (standard ONNX Gemm convention)
    w_gemm = w_data.T.flatten().tolist()
    b_data = [0.1, -0.2]
    
    W_init = helper.make_tensor('W', TensorProto.FLOAT, [2, 4], w_gemm)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [2], b_data)
    
    gemm_node = helper.make_node(
        'Gemm',
        inputs=['input', 'W', 'B'],
        outputs=['output'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )
    
    graph = helper.make_graph(
        [gemm_node],
        'linear_model',
        [X], [Y],
        [W_init, B_init]
    )
    
    model = helper.make_model(graph, producer_name='bend-onnx', ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = os.path.join(MODELS_DIR, 'linear_model.onnx')
    onnx.save(model, path)
    print(f"Saved: {path}")

def create_mlp_model():
    """2-layer MLP classifier: Input (1x4) -> Gemm (4x8) -> Relu -> Gemm (8x3) -> Softmax (1x3)"""
    np.random.seed(123)
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
    # Layer 1 weights (8 neurons, each 4 inputs)
    w1 = np.random.uniform(-0.8, 0.8, (8, 4)).astype(np.float32)
    b1 = np.random.uniform(-0.2, 0.2, (8,)).astype(np.float32)
    
    # Layer 2 weights (3 neurons, each 8 inputs)
    w2 = np.random.uniform(-0.8, 0.8, (3, 8)).astype(np.float32)
    b2 = np.random.uniform(-0.2, 0.2, (3,)).astype(np.float32)
    
    W1_init = helper.make_tensor('W1', TensorProto.FLOAT, [8, 4], w1.flatten().tolist())
    B1_init = helper.make_tensor('B1', TensorProto.FLOAT, [8], b1.flatten().tolist())
    W2_init = helper.make_tensor('W2', TensorProto.FLOAT, [3, 8], w2.flatten().tolist())
    B2_init = helper.make_tensor('B2', TensorProto.FLOAT, [3], b2.flatten().tolist())
    
    node_gemm1 = helper.make_node('Gemm', ['input', 'W1', 'B1'], ['h1_raw'], transB=1)
    node_relu1 = helper.make_node('Relu', ['h1_raw'], ['h1'])
    node_gemm2 = helper.make_node('Gemm', ['h1', 'W2', 'B2'], ['logits'], transB=1)
    node_softmax = helper.make_node('Softmax', ['logits'], ['output'], axis=1)
    
    graph = helper.make_graph(
        [node_gemm1, node_relu1, node_gemm2, node_softmax],
        'mlp_classifier',
        [X], [Y],
        [W1_init, B1_init, W2_init, B2_init]
    )
    
    model = helper.make_model(graph, producer_name='bend-onnx', ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = os.path.join(MODELS_DIR, 'mlp_classifier.onnx')
    onnx.save(model, path)
    print(f"Saved: {path}")

def create_deep_digit_classifier():
    """3-layer digit recognition network: 16 -> 32 -> 16 -> 10 classes"""
    np.random.seed(999)
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    w1 = np.random.normal(0, 0.3, (32, 16)).astype(np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    
    w2 = np.random.normal(0, 0.3, (16, 32)).astype(np.float32)
    b2 = np.zeros(16, dtype=np.float32)
    
    w3 = np.random.normal(0, 0.3, (10, 16)).astype(np.float32)
    b3 = np.zeros(10, dtype=np.float32)
    
    W1_init = helper.make_tensor('W1', TensorProto.FLOAT, [32, 16], w1.flatten().tolist())
    B1_init = helper.make_tensor('B1', TensorProto.FLOAT, [32], b1.flatten().tolist())
    W2_init = helper.make_tensor('W2', TensorProto.FLOAT, [16, 32], w2.flatten().tolist())
    B2_init = helper.make_tensor('B2', TensorProto.FLOAT, [16], b2.flatten().tolist())
    W3_init = helper.make_tensor('W3', TensorProto.FLOAT, [10, 16], w3.flatten().tolist())
    B3_init = helper.make_tensor('B3', TensorProto.FLOAT, [10], b3.flatten().tolist())
    
    node_g1 = helper.make_node('Gemm', ['input', 'W1', 'B1'], ['h1_raw'], transB=1)
    node_r1 = helper.make_node('Relu', ['h1_raw'], ['h1'])
    node_g2 = helper.make_node('Gemm', ['h1', 'W2', 'B2'], ['h2_raw'], transB=1)
    node_r2 = helper.make_node('Relu', ['h2_raw'], ['h2'])
    node_g3 = helper.make_node('Gemm', ['h2', 'W3', 'B3'], ['logits'], transB=1)
    node_sm = helper.make_node('Softmax', ['logits'], ['output'], axis=1)
    
    graph = helper.make_graph(
        [node_g1, node_r1, node_g2, node_r2, node_g3, node_sm],
        'deep_digit_classifier',
        [X], [Y],
        [W1_init, B1_init, W2_init, B2_init, W3_init, B3_init]
    )
    
    model = helper.make_model(graph, producer_name='bend-onnx', ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = os.path.join(MODELS_DIR, 'digit_classifier.onnx')
    onnx.save(model, path)
    print(f"Saved: {path}")

def create_matmul_model():
    """Pure MatMul: Input A (1x3) * B (3x2) -> Output (1x2)"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
    
    b_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [3, 2], b_data)
    
    node_mm = helper.make_node('MatMul', ['input', 'B'], ['output'])
    
    graph = helper.make_graph(
        [node_mm],
        'pure_matmul',
        [X], [Y],
        [B_init]
    )
    
    model = helper.make_model(graph, producer_name='bend-onnx', ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = os.path.join(MODELS_DIR, 'matmul_model.onnx')
    onnx.save(model, path)
    print(f"Saved: {path}")

if __name__ == '__main__':
    create_linear_model()
    create_mlp_model()
    create_deep_digit_classifier()
    create_matmul_model()
    print("All ONNX models generated successfully!")
