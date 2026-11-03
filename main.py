from src import functions as f
from src.constants import FEATURES


def main():
    design = f.select_design()
    data = f.load_data(design)
    X = data[FEATURES].to_numpy()
    y = data["mAb [mg/L]"].to_numpy()
    groups = data["Batch ID"].to_numpy()

    comparison, cv_pred = f.compare_models(X, y, groups)
    print(comparison.to_string(index=False))

    model = f.new_gp().fit(X, y)
    recommendation, best = f.optimize(model)
    print(recommendation.to_string(index=False))

    f.plot_design(design)
    f.plot_predicted_vs_actual(y, cv_pred)
    f.plot_stp_vs_time(model, best)
    f.plot_effects(model, best)


if __name__ == "__main__":
    main()
