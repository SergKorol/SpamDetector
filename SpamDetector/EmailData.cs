using Microsoft.ML.Data;

namespace SpamDetector;

public class EmailData
{
    [LoadColumn(0)] public string Sender { get; set; } = string.Empty;

    [LoadColumn(1)] public string Subject { get; set; } = string.Empty;

    [LoadColumn(2)] public string Body { get; set; } = string.Empty;

    [LoadColumn(3)] public bool IsSpam { get; set; }
}