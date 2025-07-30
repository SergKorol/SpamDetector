using Microsoft.ML;

namespace SpamDetector;

static class Program
{
    private const string DataPath = "email_dataset.tsv";
    private const string ModelPath = "spam_model.zip";
    private const string FeedbackPath = "feedback.tsv";

    static void Main()
    {
        Console.WriteLine("=== System for checking emails for spam ===\n");

        var mlContext = new MLContext();

        var pipeline = BuildPipeline(mlContext);

        ITransformer model = LoadOrTrainModel(mlContext, pipeline);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

        RunInteractiveCheck(mlContext, pipeline, ref model, ref predictionEngine);

        Console.WriteLine("The app completed successfully. Goodbye!");
    }

    private static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
    {
        return mlContext.Transforms.Text
            .FeaturizeText("SenderFeatures", nameof(EmailData.Sender))
            .Append(mlContext.Transforms.Text.FeaturizeText("SubjectFeatures", nameof(EmailData.Subject)))
            .Append(mlContext.Transforms.Text.FeaturizeText("BodyFeatures", nameof(EmailData.Body)))
            .Append(mlContext.Transforms.Concatenate("Features", "SenderFeatures", "SubjectFeatures", "BodyFeatures"))
            .Append(mlContext.Transforms.NormalizeLpNorm("Features"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(EmailData.IsSpam), featureColumnName: "Features"));
    }

    private static ITransformer LoadOrTrainModel(MLContext mlContext, IEstimator<ITransformer> pipeline)
    {
        if (File.Exists(ModelPath))
        {
            Console.WriteLine("Loading saved model...");
            return mlContext.Model.Load(ModelPath, out _);
        }

        Console.WriteLine("The model is not found. Training the new model...");
        var allData = LoadAllData(mlContext);
        return TrainEvaluateSaveModel(mlContext, pipeline, allData, saveFeedback: false);
    }

    private static List<EmailData> LoadAllData(MLContext mlContext)
    {
        IDataView originalData = mlContext.Data.LoadFromTextFile<EmailData>(
            DataPath, separatorChar: '\t', hasHeader: true);

        var allExamples = mlContext.Data
            .CreateEnumerable<EmailData>(originalData, reuseRowObject: false)
            .ToList();

        if (File.Exists(FeedbackPath))
        {
            Console.WriteLine("Found feedback data. Including it in training...");
            IDataView feedbackData = mlContext.Data.LoadFromTextFile<EmailData>(
                FeedbackPath, separatorChar: '\t', hasHeader: false);

            var feedbackList = mlContext.Data
                .CreateEnumerable<EmailData>(feedbackData, reuseRowObject: false)
                .ToList();

            allExamples.AddRange(feedbackList);
        }
        return allExamples;
    }

    private static ITransformer TrainEvaluateSaveModel(
        MLContext mlContext,
        IEstimator<ITransformer> pipeline,
        List<EmailData> allData,
        bool saveFeedback)
    {
        var allDataView = mlContext.Data.LoadFromEnumerable(allData);
        var split = mlContext.Data.TrainTestSplit(allDataView, testFraction: 0.2);

        Console.WriteLine("Training model...");
        var model = pipeline.Fit(split.TrainSet);

        Console.WriteLine("Evaluating model...");
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(EmailData.IsSpam));

        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}\n");

        mlContext.Model.Save(model, allDataView.Schema, ModelPath);
        CopyFileToProjectDirectory(ModelPath);
        if (saveFeedback)
            CopyFileToProjectDirectory(FeedbackPath);

        Console.WriteLine($"The model saved to {ModelPath}\n");
        return model;
    }

    private static void RunInteractiveCheck(
        MLContext mlContext,
        IEstimator<ITransformer> pipeline,
        ref ITransformer model,
        ref PredictionEngine<EmailData, SpamPrediction> predictionEngine)
    {
        Console.WriteLine("=== Interactive email checking ===");
        Console.WriteLine("Enter 'q' to exit\n");

        while (true)
        {
            string? sender = PromptInput("Sender email: ");
            if (sender == null) break;
            string? subject = PromptInput("Email subject: ");
            if (subject == null) break;
            string? body = PromptInput("Email body: ");
            if (body == null) break;

            var email = new EmailData { Sender = sender, Subject = subject, Body = body };
            var prediction = predictionEngine.Predict(email);

            Console.WriteLine($"\nResult: {(prediction.IsSpam ? "SPAM" : "NOT SPAM")}");
            Console.WriteLine($"Spam probability: {prediction.Probability:P2}");
            Console.WriteLine($"Score: {prediction.Score:F2}");

            var feedback = PromptInput("Do you agree with the result? (y/n): ", toLower: true);
            if (feedback == "n")
            {
                var userLabelInput = PromptInput("Does it SPAM? (y/n): ", toLower: true);
                bool userLabel = userLabelInput == "y";
                SaveFeedback(sender, subject, body, userLabel);

                Console.WriteLine("The feedback has been saved. Retraining the model with full dataset...");

                var allData = LoadAllData(mlContext);
                model = TrainEvaluateSaveModel(mlContext, pipeline, allData, saveFeedback: true);
                predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

                Console.WriteLine("The model has been retrained with full dataset and your feedback.");
            }
            Console.WriteLine(new string('-', 50) + "\n");
        }
    }

    private static string? PromptInput(string message, bool toLower = false)
    {
        Console.Write(message);
        var input = Console.ReadLine();
        if (input?.ToLower() == "q") return null;
        return toLower ? input?.ToLower() : input;
    }

    private static void SaveFeedback(string? sender, string? subject, string? body, bool userLabel)
    {
        var feedbackLine = $"{sender}\t{subject}\t{body}\t{userLabel}";
        File.AppendAllText(FeedbackPath, feedbackLine + Environment.NewLine);
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