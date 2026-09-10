// task3.c
#include <stdio.h>
#include <omp.h>

#define THREADS 10
#define ITERATIONS 1000000
#define EXPECTED 10000000L

volatile long counter = 0;
omp_lock_t lock;

void run_unlocked() {
    counter = 0;
    #pragma omp parallel num_threads(THREADS)
    {
        for (int i = 0; i < ITERATIONS; i++) {
            counter++; // Non-atomic Read-Modify-Write
        }
    }
}

void run_locked() {
    counter = 0;
    #pragma omp parallel num_threads(THREADS)
    {
        for (int i = 0; i < ITERATIONS; i++) {
            omp_set_lock(&lock);
            counter++;
            omp_unset_lock(&lock);
        }
    }
}

int main() {
    omp_init_lock(&lock);
    long results[10];

    printf("=== TASK 3: 10 UNSYNCHRONIZED RUNS ===\n");
    printf("| Run | Measured Output | Error (10^7 - Act) |\n");
    printf("|-----|-----------------|--------------------|\n");

    for (int i = 0; i < 10; i++) {
        run_unlocked();
        results[i] = counter;
        long error = EXPECTED - results[i];
        printf("| #%-2d | %-15ld | %-18ld |\n", i + 1, results[i], error);
    }

    printf("\n=== TASK 3.2: TIMING COMPARISON ===\n");

    // Time Unlocked
    double start_u = omp_get_wtime();
    run_unlocked();
    double time_unlocked_ms = (omp_get_wtime() - start_u) * 1000.0;

    // Time Locked
    double start_l = omp_get_wtime();
    run_locked();
    double time_locked_ms = (omp_get_wtime() - start_l) * 1000.0;

    printf("Unlocked: %.2f ms (Final Count: %ld)\n", time_unlocked_ms, counter);
    printf("Locked:   %.2f ms (Final Count: %ld)\n", time_locked_ms, EXPECTED);

    omp_destroy_lock(&lock);
    return 0;
}
