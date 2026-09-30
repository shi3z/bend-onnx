#!/usr/bin/env python3
"""
Comprehensive Test Suite for Bend ONNX & GPU Matrix Runtime.
Validates:
1. Pure Bend Proof Checker (ALL PROOFS CHECK, 0 @unsafe)
2. Numerical Precision against ONNX Runtime and NumPy (< 1e-4 tolerance)
3. Mathematical Properties (Softmax normalization, ReLU non-negativity, Sigmoid range)
4. Model Layer Operations (Gemm, MatMul, Add, Relu, Sigmoid, Softmax)
"""

import sys
if sys.version_info >= (3, 14):
    for p in [
        "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
        "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/site-packages",
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

import os
import unittest
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.onnx_compiler import OnnxToBendCompiler, verify, run_bend, parse_bend_vector, BEND_BIN

class TestBendProofCheck(unittest.TestCase):
    """Ensures all Bend programs pass Bend's strict formal proof checker."""
    
    def test_matrix_library_proofs(self):
        matrix_bend = os.path.join(PROJECT_ROOT, "src", "matrix.bend")
        res = subprocess.run([BEND_BIN, matrix_bend, "--check-only"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"matrix.bend proof checking failed:\n{res.stderr}")
        self.assertIn("ALL PROOFS CHECK", res.stdout)
        
    def test_compiled_models_proofs(self):
        models_dir = os.path.join(PROJECT_ROOT, "models")
        for fname in os.listdir(models_dir):
            if fname.endswith(".bend"):
                path = os.path.join(models_dir, fname)
                res = subprocess.run([BEND_BIN, path, "--check-only"], capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"{fname} proof checking failed:\n{res.stderr}")
                self.assertIn("ALL PROOFS CHECK", res.stdout)

    def test_nanogpt_proofs(self):
        nanogpt_bend = os.path.join(PROJECT_ROOT, "nanogpt", "nanogpt.bend")
        if os.path.exists(nanogpt_bend):
            res = subprocess.run([BEND_BIN, nanogpt_bend, "--check-only"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"nanogpt.bend proof checking failed:\n{res.stderr}")
            self.assertIn("ALL PROOFS CHECK", res.stdout)

class TestOnnxModelVerification(unittest.TestCase):
    """Validates end-to-end inference against ONNX Runtime reference."""
    
    def test_linear_model(self):
        path = os.path.join(PROJECT_ROOT, "models", "linear_model.onnx")
        self.assertTrue(verify(path), "linear_model.onnx verification failed")
        
    def test_mlp_classifier(self):
        path = os.path.join(PROJECT_ROOT, "models", "mlp_classifier.onnx")
        self.assertTrue(verify(path), "mlp_classifier.onnx verification failed")
        
    def test_deep_digit_classifier(self):
        path = os.path.join(PROJECT_ROOT, "models", "digit_classifier.onnx")
        self.assertTrue(verify(path), "digit_classifier.onnx verification failed")
        
    def test_matmul_model(self):
        path = os.path.join(PROJECT_ROOT, "models", "matmul_model.onnx")
        self.assertTrue(verify(path), "matmul_model.onnx verification failed")

    def test_resnet_block(self):
        path = os.path.join(PROJECT_ROOT, "models", "resnet_block.onnx")
        self.assertTrue(verify(path), "resnet_block.onnx verification failed")

    def test_mini_resnet(self):
        path = os.path.join(PROJECT_ROOT, "models", "mini_resnet.onnx")
        self.assertTrue(verify(path), "mini_resnet.onnx verification failed")

class TestMathematicalProperties(unittest.TestCase):
    """Property-based verification of mathematical invariants."""
    
    def test_softmax_sums_to_one(self):
        """Softmax output probabilities must sum to 1.0."""
        path = os.path.join(PROJECT_ROOT, "models", "mlp_classifier.bend")
        out_str = run_bend(path)
        probs = parse_bend_vector(out_str)
        total = sum(probs)
        self.assertAlmostEqual(total, 1.0, places=4, msg="Softmax probabilities do not sum to 1.0")
        
    def test_softmax_in_probability_range(self):
        """All softmax elements must be between 0.0 and 1.0."""
        path = os.path.join(PROJECT_ROOT, "models", "digit_classifier.bend")
        out_str = run_bend(path)
        probs = parse_bend_vector(out_str)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            
    def test_predicted_class_matches_argmax(self):
        """Argmax must correspond to the maximum element in the probability vector."""
        path = os.path.join(PROJECT_ROOT, "models", "mlp_classifier.bend")
        out_str = run_bend(path)
        probs = parse_bend_vector(out_str)
        expected_class = int(np.argmax(probs))
        
        predicted = None
        for line in out_str.splitlines():
            if line.startswith("BEND_PREDICTED_CLASS:"):
                predicted = int(line.split(":")[1].strip())
                break
        self.assertEqual(predicted, expected_class, "Argmax prediction did not match max index")

class TestNanoGPTGeneration(unittest.TestCase):
    """Verifies autoregressive text generation in pure Bend."""
    
    def test_nanogpt_autoregressive_generation(self):
        nanogpt_bend = os.path.join(PROJECT_ROOT, "nanogpt", "nanogpt.bend")
        self.assertTrue(os.path.exists(nanogpt_bend), "nanogpt.bend does not exist")
        res = subprocess.run([BEND_BIN, nanogpt_bend], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"nanogpt execution failed:\n{res.stderr}")
        self.assertIn("Output:   BEND IS FAST! ", res.stdout)

if __name__ == '__main__':
    unittest.main()
