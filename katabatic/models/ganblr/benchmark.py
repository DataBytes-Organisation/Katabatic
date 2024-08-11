import time
import numpy as np
from ganblr import GANBLR as OriginalGANBLR
from ganblr_optimized import GANBLR as OptimizedGANBLR

def benchmark(model_class, X, y, n_runs=5):
    total_fit_time = 0
    total_sample_time = 0
    total_evaluate_time = 0

    for _ in range(n_runs):
        model = model_class()

        start_time = time.time()
        model.fit(X, y, epochs=1)
        total_fit_time += time.time() - start_time

        start_time = time.time()
        model.sample(1000)
        total_sample_time += time.time() - start_time

        start_time = time.time()
        model.evaluate(X, y)
        total_evaluate_time += time.time() - start_time

    return {
        "fit_time": total_fit_time / n_runs,
        "sample_time": total_sample_time / n_runs,
        "evaluate_time": total_evaluate_time / n_runs
    }

if __name__ == "__main__":
    X = np.random.randint(0, 5, size=(1000, 10))
    y = np.random.randint(0, 2, size=1000)

    original_results = benchmark(OriginalGANBLR, X, y)
    optimized_results = benchmark(OptimizedGANBLR, X, y)

    print("Original GANBLR:")
    print(original_results)
    print("\nOptimized GANBLR:")
    print(optimized_results)

    for key in original_results:
        improvement = (original_results[key] - optimized_results[key]) / original_results[key] * 100
        print(f"\n{key} improvement: {improvement:.2f}%")
