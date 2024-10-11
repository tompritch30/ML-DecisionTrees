#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double accuracy;
    int n_estimators;
    int tree_depth;
    int min_samples_split;
    int max_features;
} Hyperparameters;

#ifdef __cplusplus
}
#endif
