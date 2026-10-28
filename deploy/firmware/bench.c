/* Benchmark harness skeleton for the end-to-end measurements of Section 3.12.
 * Target: Zephyr (Cortex-M55 + Ethos-U55 and RISC-V U74).  Fill the three HOOK functions with the exported models
 * (deploy/export_models.py output: cnn.tflite, bilstm.tflite, lgbm_model.c, hfr_extract.h).
 *
 * What it does per decision (>= 10,000 decisions per configuration):
 *   1. replays a stored window of raw packets/feature rows from flash (so every run sees identical input),
 *   2. times capture-buffering, HFR extraction (incl. 28x10 entropy histogram), scaling, CNN, BiLSTM, LightGBM and DWF with the
 *      core cycle counter (DWT->CYCCNT on Cortex-M, rdcycle on RISC-V),
 *   3. raises a GPIO marker (stage id 1..7) for the current profiler, so energy per stage can be integrated afterwards,
 *   4. prints "decision_id,stage,cycles,cpu_hz" lines on UART (stages_<board>.csv), and records the stack/arena high-water marks.
 * Run the profiler export through deploy/ingest.py.  Flash comes from the linker map (size of .text/.rodata per component),
 * peak SRAM from the high-water marks printed at the end.
 */
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <stdio.h>
#include <stdint.h>
#include "hfr_extract.h"

#if defined(CONFIG_CPU_CORTEX_M)
  #define CYC_INIT()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while (0)
  #define CYC_NOW()   (DWT->CYCCNT)
#else
  static inline uint32_t rd_cycle(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return (uint32_t)c; }
  #define CYC_INIT()  do {} while (0)
  #define CYC_NOW()   rd_cycle()
#endif

#define N_DECISIONS 10000
#define MARK(id) mark_gpio(id)
extern void mark_gpio(int id);                 /* drive 3 GPIO lines = stage id for the current profiler */

/* ---- HOOKS: implement with the exported models ---- */
extern void hook_capture(int decision);                 /* copy window of T packets/rows into the ring buffer        */
extern void hook_hfr(void);                             /* 28 HFR components + 28x10 histogram, writes float x[28]    */
extern void hook_scale(void);                           /* (x-med)/iqr, clip, int8 quantise                           */
extern void hook_cnn(void);  extern void hook_bilstm(void);  extern void hook_lgbm(void);
extern int  hook_dwf(void);                             /* entropy -> regime -> weights -> fused argmax               */
extern uint32_t stack_hwm_bytes(void); extern uint32_t arena_hwm_bytes(void);

typedef void (*fn_t)(void);
static void cap_wrap(int d) { hook_capture(d); }

int main(void) {
    CYC_INIT();
    printk("decision_id,stage,cycles,cpu_hz\n");
    const uint32_t hz = sys_clock_hw_cycles_per_sec();
    for (int d = 0; d < N_DECISIONS; ++d) {
        uint32_t t0, t1, tt0 = CYC_NOW();
        MARK(1); t0 = CYC_NOW(); cap_wrap(d);   t1 = CYC_NOW(); printk("%d,capture,%u,%u\n", d, t1 - t0, hz);
        MARK(2); t0 = CYC_NOW(); hook_hfr();    t1 = CYC_NOW(); printk("%d,hfr,%u,%u\n",     d, t1 - t0, hz);
        MARK(3); t0 = CYC_NOW(); hook_scale();  t1 = CYC_NOW(); printk("%d,scale,%u,%u\n",   d, t1 - t0, hz);
        MARK(4); t0 = CYC_NOW(); hook_cnn();    t1 = CYC_NOW(); printk("%d,cnn,%u,%u\n",     d, t1 - t0, hz);
        MARK(5); t0 = CYC_NOW(); hook_bilstm(); t1 = CYC_NOW(); printk("%d,bilstm,%u,%u\n",  d, t1 - t0, hz);
        MARK(6); t0 = CYC_NOW(); hook_lgbm();   t1 = CYC_NOW(); printk("%d,lgbm,%u,%u\n",    d, t1 - t0, hz);
        MARK(7); t0 = CYC_NOW(); (void)hook_dwf(); t1 = CYC_NOW(); printk("%d,dwf,%u,%u\n",   d, t1 - t0, hz);
        MARK(0); printk("%d,total,%u,%u\n", d, CYC_NOW() - tt0, hz);
    }
    printk("# stack_hwm_bytes=%u arena_hwm_bytes=%u\n", stack_hwm_bytes(), arena_hwm_bytes());
    return 0;
}
