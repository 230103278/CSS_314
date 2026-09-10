#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

// Compute-bound workload: prime test via trial division
static inline bool is_prime(long n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

long run_benchmark(long limit, int num_threads) {
    long prime_count = 0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1000) reduction(+:prime_count)
    for (long i = 2; i <= limit; i++) {
        if (is_prime(i)) {
            prime_count++;
        }
    }
    return prime_count;
}

int main() {
    const long LIMIT = 20000000; // 20 Million
    const int thread_counts[] = {1, 2, 4, 8, 16, 32};
    const int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    const int RUNS = 3;

    double t1_avg = 0.0;

    printf("\n=== RUNNING TASK 2 BENCHMARK (AMD Zen 4) ===\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-12s | %-12s | %-12s\n",
           "Threads", "Run 1 (s)", "Run 2 (s)", "Run 3 (s)", "Avg TN (s)", "Speedup SN", "Efficiency EN");
    printf("-----------------------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        double times[3];
        double sum = 0.0;

        for (int r = 0; r < RUNS; r++) {
            double start = omp_get_wtime();
            run_benchmark(LIMIT, threads);
            double end = omp_get_wtime();
            times[r] = end - start;
            sum += times[r];
        }

        double avg = sum / RUNS;
        if (threads == 1) {
            t1_avg = avg;
        }

        double speedup = t1_avg / avg;
        double efficiency = (speedup / threads) * 100.0;

        printf("%-8d | %-10.4f | %-10.4f | %-10.4f | %-12.4f | %-12.2fx | %-11.1f%%\n",
               threads, times[0], times[1], times[2], avg, speedup, efficiency);
    }

    return 0;
}
