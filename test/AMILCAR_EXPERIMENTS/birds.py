import csv

import pandas as pd
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

from src.selection.select import Selection
from src.utils.general_utils import Utilities
from src.utils.test_utils import TestUtils


def train_and_evaluate():
    print("Importing dataset.")
    # Get the dataset ready for augmentation procedure
    birds = PTRAILDataFrame(data_set=pd.read_csv('../datasets/birds.csv'),
                            traj_id='traj_id',
                            datetime='DateTime',
                            latitude='lat',
                            longitude='lon')
    ready_dataset = KinematicFeatures.create_distance_column(birds)
    print("Dataset ready for testing and augmentation procedure.")

    seed_generator = Utilities.generate_pi_seed(20)
    seed_vals = [next(seed_generator) for i in range(20)]
    shake_percentages = [0.2, 0.4, 0.6]
    circle_methods = ['on', 'in']
    ml_models = [ExtraTreesClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]
    scaler = MinMaxScaler((0, 1))

    distance_results = [["seed", "on_20%_dist", "on_20%_std", "on_40%_dist", "on_40%_std", "on_60%_dist", "on_60%_std",
                         "in_20%_dist", "in_20%_std", "in_40%_dist", "in_40%_std", "in_60%_dist", "in_60%_std"]]

    model_results = [
        ["seed", "model", "baseline", "in_20%_f1", "in_40%_f1", "in_60%_f1", "on_20%_f1", "on_40%_f1", "on_60%_f1"]]

    for seed in seed_vals:
        # Intermediate lists for storing distance and model score values.
        distance_row = [seed]

        # Set apart 20% data for testing that augmentation process will never see.
        train, test_x, test_y = TestUtils.get_test_train_data(dataset=ready_dataset, seed_val=seed,
                                                              class_col='Species', k=0.8)

        model_row = TestUtils.create_model_row(seed, ml_models, "Species", train, test_x, test_y)
        for shake in shake_percentages:
            for method in circle_methods:
                # Randomly select 30% of trajectories to be augmented.
                selected = Selection.select_randomly(train, seed, k=0.3)

                # Augment the trajectories.
                train_x, train_y = TestUtils.augment_trajectories_using_random_strategy(dataset=train,
                                                                                        percent_to_shake=shake,
                                                                                        ids_to_augment=selected,
                                                                                        circle=method,
                                                                                        n_augmentations=20,
                                                                                        class_col="Species")
                mean, std = TestUtils.find_original_and_augmentation_pairs_and_calculate_differences(train_x, selected)
                distance_row.append(mean)
                distance_row.append(std)

                for i in range(len(ml_models)):
                    f1_score = TestUtils.train_model_and_evaluate(ml_models[i], scaler.fit_transform(train_x), train_y,
                                                                  scaler.fit_transform(test_x), test_y, seed)
                    model_row[i].append(f1_score)

        model_results.extend(model_row)
        distance_results.append(distance_row)

        print(model_row)

    file_path = "results/bird_distances.csv"
    with open(file_path, mode="w") as file:
        writer = csv.writer(file)
        for item in distance_results:
            writer.writerow(item)
        print(f"File successfully written to: {file_path}")

    file_path = "results/bird_f1_score.csv"
    with open(file_path, mode="w") as file:
        writer = csv.writer(file)
        for item in model_results:
            writer.writerow(item)
        print(f"File successfully written to: {file_path}")


if __name__ == "__main__":
    train_and_evaluate()
