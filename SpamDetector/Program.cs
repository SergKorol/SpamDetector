using Microsoft.ML;

namespace SpamDetector;

static class Program
{
    private const string DataPath = "email_dataset.tsv";
    private const string ModelPath = "spam_model.zip";
    private const string FeedbackPath = "feedback.tsv";

    static void Main()
    {
        Console.WriteLine("=== System for checking emails for spam ===");
        Console.WriteLine();

        var mlContext = new MLContext();

        var pipeline = mlContext.Transforms.Text
            .FeaturizeText(outputColumnName: "SenderFeatures", inputColumnName: nameof(EmailData.Sender))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "SubjectFeatures",
                inputColumnName: nameof(EmailData.Subject)))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "BodyFeatures",
                inputColumnName: nameof(EmailData.Body)))
            .Append(mlContext.Transforms.Concatenate("Features", "SenderFeatures", "SubjectFeatures", "BodyFeatures"))
            .Append(mlContext.Transforms.NormalizeLpNorm("Features"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(EmailData.IsSpam), featureColumnName: "Features"));

        ITransformer model;
        if (File.Exists(ModelPath))
        {
            Console.WriteLine("Loading saved model...");
            model = mlContext.Model.Load(ModelPath, out _);
        }
        // else
        // {
        //     Console.WriteLine("The model is not found. Training the new model...");
        //     IDataView dataView = mlContext.Data.LoadFromTextFile<EmailData>(
        //         DataPath,
        //         separatorChar: '\t',
        //         hasHeader: true);
        //
        //     var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        //
        //     Console.WriteLine("Training model...");
        //     model = pipeline.Fit(split.TrainSet);
        //
        //     Console.WriteLine("Evaluating model...");
        //     var predictions = model.Transform(split.TestSet);
        //     var metrics =
        //         mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(EmailData.IsSpam));
        //
        //     Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        //     Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        //     Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        //     Console.WriteLine();
        //
        //     mlContext.Model.Save(model, dataView.Schema, ModelPath);
        //     CopyFileToProjectDirectory(ModelPath);
        //     Console.WriteLine($"The model saved to {ModelPath}");
        //     Console.WriteLine();
        // }
        else
        {
            Console.WriteLine("The model is not found. Training the new model...");

            IDataView originalData = mlContext.Data.LoadFromTextFile<EmailData>(
                DataPath,
                separatorChar: '\t',
                hasHeader: true);

            List<EmailData> allExamples = mlContext.Data
                .CreateEnumerable<EmailData>(originalData, reuseRowObject: false)
                .ToList();

            if (File.Exists(FeedbackPath))
            {
                Console.WriteLine("Found feedback data. Including it in training...");
                IDataView feedbackData = mlContext.Data.LoadFromTextFile<EmailData>(
                    FeedbackPath,
                    separatorChar: '\t',
                    hasHeader: false); // Feedback file has no header

                var feedbackList = mlContext.Data
                    .CreateEnumerable<EmailData>(feedbackData, reuseRowObject: false)
                    .ToList();

                allExamples.AddRange(feedbackList);
            }

            var allDataView = mlContext.Data.LoadFromEnumerable(allExamples);

            var split = mlContext.Data.TrainTestSplit(allDataView, testFraction: 0.2);

            Console.WriteLine("Training model...");
            model = pipeline.Fit(split.TrainSet);

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(EmailData.IsSpam));

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine();

            mlContext.Model.Save(model, allDataView.Schema, ModelPath);
            CopyFileToProjectDirectory(ModelPath);
            Console.WriteLine($"The model saved to {ModelPath}");
            Console.WriteLine();
        }

        var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

        Console.WriteLine("=== Interactive email checking ===");
        Console.WriteLine("Enter 'q' to exit");
        Console.WriteLine();

        while (true)
        {
            Console.Write("Sender email: ");
            var sender = Console.ReadLine();
            if (sender?.ToLower() == "q") break;

            Console.Write("Email subject: ");
            var subject = Console.ReadLine();
            if (subject?.ToLower() == "q") break;

            Console.Write("Email body: ");
            var body = Console.ReadLine();
            if (body?.ToLower() == "q") break;

            var email = new EmailData { Sender = sender ?? "", Subject = subject ?? "", Body = body ?? "" };
            var prediction = predictionEngine.Predict(email);

            Console.WriteLine();
            Console.WriteLine($"Result: {(prediction.IsSpam ? "SPAM" : "NOT SPAM")}");
            Console.WriteLine($"Spam probability: {prediction.Probability:P2}");
            Console.WriteLine($"Score: {prediction.Score:F2}");

            Console.Write("Do you agree with the result? (y/n): ");
            var feedback = Console.ReadLine()?.ToLower();
            
            if (feedback == "n")
            {
                Console.Write("Does it SPAM? (y/n): ");
                var userLabel = Console.ReadLine()?.ToLower() == "y";

                var feedbackLine = $"{sender}\t{subject}\t{body}\t{userLabel}";
                File.AppendAllText("feedback.tsv", feedbackLine + Environment.NewLine);

                Console.WriteLine("The feedback has been saved. Retraining the model with full dataset...");

                // Load original dataset
                IDataView originalData = mlContext.Data.LoadFromTextFile<EmailData>(
                    DataPath,
                    separatorChar: '\t',
                    hasHeader: true);

                // Load feedback dataset (no header)
                IDataView feedbackData = mlContext.Data.LoadFromTextFile<EmailData>(
                    FeedbackPath,
                    separatorChar: '\t',
                    hasHeader: false);

                // Convert both to IEnumerable<EmailData>
                var originalList = mlContext.Data.CreateEnumerable<EmailData>(originalData, reuseRowObject: false).ToList();
                var feedbackList = mlContext.Data.CreateEnumerable<EmailData>(feedbackData, reuseRowObject: false).ToList();

                // Combine both
                var combinedData = originalList.Concat(feedbackList);

                // Load into IDataView
                var allDataView = mlContext.Data.LoadFromEnumerable(combinedData);

                // Train/test split
                var split = mlContext.Data.TrainTestSplit(allDataView, testFraction: 0.2);
                model = pipeline.Fit(split.TrainSet);

                // Evaluate
                var predictions = model.Transform(split.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(EmailData.IsSpam));
                Console.WriteLine($"New Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"New AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"New F1 Score: {metrics.F1Score:P2}");

                // Save updated model
                mlContext.Model.Save(model, allDataView.Schema, ModelPath);
                CopyFileToProjectDirectory(ModelPath);
                CopyFileToProjectDirectory(FeedbackPath);
                predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

                Console.WriteLine("The model has been retrained with full dataset and your feedback.");
            }

            Console.WriteLine(new string('-', 50));
            Console.WriteLine();
        }

        Console.WriteLine("The app completed successfully. Goodbye!");
    }

    private static void CopyFileToProjectDirectory(string fileName)
    {
        string currentDir = Directory.GetCurrentDirectory();
        string projectDir = Path.GetFullPath(Path.Combine(currentDir, "..", "..", ".."));
        string sourcePath = Path.Combine(currentDir, fileName);
        string destPath = Path.Combine(projectDir, fileName);
            
        File.Copy(sourcePath, destPath, overwrite: true);
    }
}
