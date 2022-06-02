from argparse import ArgumentParser
from mlds6sentiment.database import VidhyaSentimentsDataLoader
from mlds6sentiment.models import ModelProxy
from mlds6sentiment.types.models import ModelKind
from mlds6sentiment.types.feature_extraction import FeatureExtractionFields
from mlds6sentiment.feature_extraction import VidhyaSentimentsFeatureExtractor
from mlds6sentiment.types.metrics import MetricsKind, ClassificationMetricsEnum
from mlds6sentiment.metrics import ClassificationMetricsCalculator
import mlflow
import mlflow.sklearn

def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--C", type=float)
    parser.add_argument("--gamma", type=float)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    dataset = (
            VidhyaSentimentsDataLoader()
            .add_path("/home/juselara/Downloads/train.csv")
            .load()
            )
    data = dataset.get_data()

    extractor = (
            VidhyaSentimentsFeatureExtractor()
            .add_data(data)
            .add_fields(
                FeatureExtractionFields(
                    label="label",
                    text="tweet"
                    )
                )
            )
    metrics_kind = MetricsKind(accuracy=True, f1_score=True)
    metrics_calculator = ClassificationMetricsCalculator(metrics_kind)

    model_proxy = (
            ModelProxy()
            .add_model_kind(ModelKind.SVM)
            .add_hparams(kernel=args.kernel, C=args.C, gamma=args.gamma)
            )

    with mlflow.start_run():
        mlflow.log_param("kernel", args.kernel)
        mlflow.log_param("C", args.C)
        mlflow.log_param("gamma", args.gamma)

        model = model_proxy.resolve()

        X, y = extractor.extract()
        extractor.save("/home/juselara/Downloads/features")
        mlflow.log_artifact("/home/juselara/Downloads/features_x.npy")
        mlflow.log_artifact("/home/juselara/Downloads/features_y.npy")

        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = metrics_calculator.compute(y, y_pred)

        for metric in ClassificationMetricsEnum:
            metric_name = metric.name.lower()
            value = metrics.dict()[metric_name]
            if value is not None:
                mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(model, "vihdya_svm")
