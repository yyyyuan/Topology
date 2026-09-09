// ===============
// Complete data flow
// ===============
// [200 Frames x 256 Patches x 512 Words]  (Raw 104.85 MB Simulation Buffer)
//              │
//              ▼  Slice into T=4 Tubelets (50 steps x 256 patches x 2,048 words)
// [50 Steps x 256 Tubelets x 2,048 Words]
//              │
//              ▼  1. Spatiotemporal Projection Kernel (W_proj: [2048, 128])
// [50 Steps x 256 Tokens x 128 Features]
//              │
//              ▼  2. Spatial Aggregation (global_avg_pool_kernel across 256 patches)
// [50 Temporal Tokens x 128 Features]
//              │
//              ▼  3. Temporal Self-Attention (Q * K^T generates [50, 50] map)
// [50 Temporal Tokens x 128 Features]
//              │
//              ▼  4. Temporal Aggregation (Average across 50 temporal steps)
// [1 Pooled Sequence Vector x 128 Features]
//              │
//              ▼  5. Linear Classifier (W_class: [128, 1000])
// [1000 Class Logits]
// ==============