import { useState, useMemo, useEffect, useRef } from "react";

// Define types for our state to improve code quality and catch errors
type ActiveTab = "upload" | "link" | "text";
type VerificationState = "idle" | "loading" | "complete";

// Type for the AI Image Detection result
interface AnalysisDetails {
  confidence: number;
  reasons: { icon: "check" | "x"; text: string }[];
  modelPrediction: {
    prediction: string;
    confidence: number;
  };
}

// Type for the Fact-Check result
interface FactCheckResult {
  source_content: {
    title: string;
    url: string;
    domain_info: {
      domain: string;
      age_days: number;
      country: string;
      registrar: string;
      is_news_domain: boolean;
      is_academic_domain: boolean;
      is_government_domain: boolean;
    };
    content_quality: {
      word_count: number;
      sentence_count: number;
      avg_sentence_length: number;
      quality_score: number;
      has_quotes: boolean;
      has_sources: boolean;
      has_numbers: boolean;
    };
    atomic_claims: string[];
  };
  verdict_analysis: {
    primary_claim: string;
    verdict: string;
    confidence_score: number;
    explanation: string;
    evidence_summary: string;
    risk_factors: string[];
    verification_suggestions: string[];
    credibility_breakdown: {
      evidence_consistency: number;
      overall_evidence_strength: number;
      source_credibility_average: number;
      source_diversity: number;
    };
  };
  evidence_analysis: {
    total_sources: number;
    sources_by_type: {
      fact_check: number;
      academic: number;
      news: number;
      official: number;
    };
    average_credibility: number;
  };
  processing_metadata: {
    processing_time_seconds: number;
    timestamp: string;
    version: string;
  };
}

// --- SVG Icon Components ---
// It's good practice to define icons as reusable components

const LogoIcon = () => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="text-[#0A4D68]"
  >
    <path
      d="M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm-3.37-2.03 7.07-7.07"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="m14.83 14.83 4.24 4.24"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="m14 10 2 2 4-4"
      stroke="#2ECC71"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const UploadIcon = () => (
  <svg
    className="w-8 h-8 mb-4 text-gray-500"
    aria-hidden="true"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 20 16"
  >
    <path
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
    />
  </svg>
);

const VerdictIcon = ({ verdict }: { verdict: string }) => {
  // Green verdicts (likely true)
  if (verdict.includes("True") || verdict.includes("Authentic")) {
    return (
      <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
        <svg
          className="w-6 h-6 text-green-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M5 13l4 4L19 7"
          />
        </svg>
      </div>
    );
  }

  // Red verdicts (likely false)
  if (verdict.includes("False") || verdict.includes("AI Detected")) {
    return (
      <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
        <svg
          className="w-6 h-6 text-red-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </div>
    );
  }

  // Yellow verdicts (uncertain/mixed)
  if (
    verdict.includes("Uncertain") ||
    verdict.includes("Inconclusive") ||
    verdict.includes("Misleading") ||
    verdict.includes("Contested") ||
    verdict.includes("Unproven") ||
    verdict.includes("Insufficient")
  ) {
    return (
      <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
        <svg
          className="w-6 h-6 text-yellow-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </div>
    );
  }

  // Default gray for unknown verdicts
  return (
    <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
      <svg
        className="w-6 h-6 text-gray-600"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    </div>
  );
};

export default function App() {
  // State for the active tab ('upload', 'link', 'text')
  const [activeTab, setActiveTab] = useState<ActiveTab>("upload");

  // State for file, link, and text inputs
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [linkUrl, setLinkUrl] = useState("");
  const [pastedText, setPastedText] = useState("");

  // State to manage the overall process flow
  const [verificationState, setVerificationState] =
    useState<VerificationState>("idle");

  // States to hold the final results from the backend
  const [analysisDetails, setAnalysisDetails] =
    useState<AnalysisDetails | null>(null);
  const [factCheckResult, setFactCheckResult] =
    useState<FactCheckResult | null>(null);

  // States for the loading animation
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);
  const loadingMessages = [
    "Analyzing...",
    "Scanning for digital artifacts...",
    "Cross-referencing sources...",
    "Finalizing report...",
  ];

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Effect to cycle through loading messages
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (verificationState === "loading") {
      interval = setInterval(() => {
        setLoadingMessageIndex(
          (prevIndex) => (prevIndex + 1) % loadingMessages.length
        );
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [loadingMessages.length, verificationState]);

  // A derived state to check if there's any content ready to be verified
  const hasContent = useMemo(() => {
    if (activeTab === "upload") return !!uploadedFile;
    if (activeTab === "link") return linkUrl.trim() !== "";
    if (activeTab === "text") return pastedText.trim() !== "";
    return false;
  }, [activeTab, uploadedFile, linkUrl, pastedText]);

  // Function to reset all inputs and results
  const resetState = () => {
    setUploadedFile(null);
    setLinkUrl("");
    setPastedText("");
    setVerificationState("idle");
    setAnalysisDetails(null);
    setFactCheckResult(null);
  };

  const handleVerify = async () => {
    if (!hasContent) return;
    setVerificationState("loading");
    setLoadingMessageIndex(0);

    // Clear previous results before starting a new analysis
    setAnalysisDetails(null);
    setFactCheckResult(null);

    let endpoint = "";
    let requestBody;
    const headers: HeadersInit = {};

    // Determine the correct endpoint and body based on the active tab
    if (activeTab === "upload" && uploadedFile) {
      endpoint = "http://127.0.0.1:5001/api/v1/detect/image";
      const formData = new FormData();
      formData.append("image", uploadedFile);
      requestBody = formData;
    } else if (activeTab === "link" && linkUrl) {
      endpoint = "http://127.0.0.1:5001/api/v1/fact-check/url";
      requestBody = JSON.stringify({ url: linkUrl });
      headers["Content-Type"] = "application/json";
    } else {
      // Placeholder for text verification
      alert("Text verification is not yet implemented.");
      setVerificationState("idle");
      return;
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: requestBody,
        headers: headers,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error ||
            `Network response was not ok: ${response.statusText}`
        );
      }

      const data = await response.json();

      // Check the structure of the response to determine how to set the state
      if (data.verdict_analysis) {
        // This is a fact-check response
        setFactCheckResult(data as FactCheckResult);
      } else if (data.details && data.details.model_top_prediction) {
        // This is an AI detection response
        setAnalysisDetails({
          confidence: Math.round(data.confidence * 100),
          reasons: [
            {
              icon: data.is_likely_ai ? "x" : "check",
              text: data.details.reason,
            },
          ],
          modelPrediction: {
            prediction: data.details.model_top_prediction.class_name,
            confidence: Math.round(
              data.details.model_top_prediction.confidence_score * 100
            ),
          },
        });
      }
    } catch (error) {
      console.error("Error during verification:", error);
      alert(
        `An error occurred: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setVerificationState("complete");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const renderVerdictTitle = () => {
    if (analysisDetails) {
      return analysisDetails.reasons[0].icon === "x"
        ? "AI Generated / Altered"
        : "Likely Authentic";
    }
    if (factCheckResult) {
      return factCheckResult.verdict_analysis.verdict;
    }
    return "Analysis Complete";
  };

  return (
    <div className="min-h-screen bg-[#F7F7F7] font-sans text-[#333333]">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="flex justify-between items-center mb-12">
          <div className="flex items-center gap-3">
            <LogoIcon />
            <span
              className="text-2xl font-bold text-[#0A4D68]"
              style={{ fontFamily: "Poppins, sans-serif" }}
            >
              VerifAI
            </span>
          </div>
          <nav className="flex gap-6 text-gray-600">
            <a href="#" className="hover:text-[#088395]">
              About
            </a>
            <a href="#" className="hover:text-[#088395]">
              How it Works
            </a>
          </nav>
        </header>

        <main className="text-center">
          {verificationState === "idle" && (
            <div className="animate-fade-in">
              <h1
                className="text-4xl md:text-5xl font-bold mb-4"
                style={{ fontFamily: "Poppins, sans-serif" }}
              >
                AI-Powered Content Verification
              </h1>
              <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
                Analyze images and links to detect AI generation and verify
                factual claims.
              </p>

              <div className="bg-white p-6 rounded-xl shadow-lg max-w-2xl mx-auto">
                <div className="flex border-b mb-4">
                  {(["upload", "link", "text"] as ActiveTab[]).map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={`py-2 px-4 font-semibold capitalize transition-colors duration-300 ${
                        activeTab === tab
                          ? "border-b-2 border-[#088395] text-[#088395]"
                          : "text-gray-500 hover:text-[#0A4D68]"
                      }`}
                    >
                      {tab === "upload"
                        ? "Upload File"
                        : tab === "link"
                        ? "Paste Link"
                        : "Paste Text"}
                    </button>
                  ))}
                </div>

                {/* Input Areas */}
                {activeTab === "upload" && (
                  <div>
                    <label
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={handleDrop}
                      className="flex justify-center w-full h-32 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-md appearance-none cursor-pointer hover:border-gray-400 focus:outline-none"
                    >
                      <span className="flex items-center space-x-2">
                        <UploadIcon />
                        <span className="font-medium text-gray-600">
                          {uploadedFile
                            ? `File: ${uploadedFile.name}`
                            : "Drop files to Attach, or "}
                          <span
                            className="text-[#088395] underline"
                            onClick={() => fileInputRef.current?.click()}
                          >
                            browse
                          </span>
                        </span>
                      </span>
                      <input
                        type="file"
                        name="file_upload"
                        className="hidden"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                      />
                    </label>
                    <p className="text-xs text-gray-500 mt-2">
                      Supports: JPG, PNG
                    </p>
                  </div>
                )}
                {activeTab === "link" && (
                  <input
                    type="text"
                    value={linkUrl}
                    onChange={(e) => setLinkUrl(e.target.value)}
                    placeholder="https://example.com/article-or-image.jpg"
                    className="w-full p-3 border rounded-md focus:ring-2 focus:ring-[#088395]"
                  />
                )}
                {activeTab === "text" && (
                  <textarea
                    value={pastedText}
                    onChange={(e) => setPastedText(e.target.value)}
                    placeholder="Copy and paste text from an article or post..."
                    className="w-full p-3 border rounded-md h-32 focus:ring-2 focus:ring-[#088395]"
                  ></textarea>
                )}
              </div>

              <button
                onClick={handleVerify}
                disabled={!hasContent}
                className="mt-8 px-8 py-3 bg-[#088395] text-white font-bold rounded-lg shadow-md hover:bg-[#0A4D68] disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
              >
                Verify
              </button>
            </div>
          )}

          {/* Loading State */}
          {verificationState === "loading" && (
            <div className="flex flex-col items-center justify-center h-64 animate-fade-in">
              <div className="animate-pulse">
                <LogoIcon />
              </div>
              <p className="mt-4 text-lg text-gray-600 transition-opacity duration-500">
                {loadingMessages[loadingMessageIndex]}
              </p>
            </div>
          )}

          {/* Results Display */}
          {verificationState === "complete" && (
            <div className="text-left animate-fade-in">
              <div className="bg-white rounded-xl shadow-lg p-8">
                <div className="flex items-center gap-4 mb-4">
                  <VerdictIcon verdict={renderVerdictTitle()} />
                  <div>
                    <h2 className="text-sm text-gray-500">Result</h2>
                    <p
                      className="text-2xl font-bold"
                      style={{ fontFamily: "Poppins, sans-serif" }}
                    >
                      {renderVerdictTitle()}
                    </p>
                  </div>
                </div>

                {/* AI Detection Result Card */}
                {analysisDetails && (
                  <div>
                    <p className="text-gray-600 mb-6">
                      Our model analyzed the image for digital artifacts and
                      compared its patterns against known real-world photos.
                    </p>
                    <div className="space-y-4">
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h3 className="font-semibold">Authenticity Analysis</h3>
                        <p className="text-gray-600">
                          {analysisDetails.reasons[0].text}
                        </p>
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h3 className="font-semibold">Model Prediction</h3>
                        <p className="text-gray-600">
                          The model identified the image content as{" "}
                          <span className="font-bold">
                            {analysisDetails.modelPrediction.prediction}
                          </span>{" "}
                          with{" "}
                          <span className="font-bold">
                            {analysisDetails.modelPrediction.confidence}%
                          </span>{" "}
                          confidence.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Fact Check Result Card */}
                {factCheckResult && (
                  <div>
                    <div className="mb-6">
                      <p className="text-gray-600 mb-2">
                        Based on the claim:{" "}
                        <span className="font-semibold italic">
                          "{factCheckResult.verdict_analysis.primary_claim}"
                        </span>
                      </p>
                      <p className="text-lg mb-4">
                        <span className="font-bold">Verdict:</span>{" "}
                        {factCheckResult.verdict_analysis.verdict}
                      </p>
                      <p className="text-lg mb-4">
                        <span className="font-bold">Confidence:</span>{" "}
                        {Math.round(
                          factCheckResult.verdict_analysis.confidence_score *
                            100
                        )}
                        %
                      </p>
                      <p className="text-gray-700 mb-4">
                        {factCheckResult.verdict_analysis.explanation}
                      </p>
                    </div>

                    {/* Source Analysis */}
                    <div className="border-t pt-4 mb-6">
                      <h3 className="text-lg font-semibold mb-3">
                        Source Analysis
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div className="p-3 bg-blue-50 rounded-lg text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {
                              factCheckResult.evidence_analysis.sources_by_type
                                .fact_check
                            }
                          </div>
                          <div className="text-sm text-gray-600">
                            Fact Checkers
                          </div>
                        </div>
                        <div className="p-3 bg-green-50 rounded-lg text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {
                              factCheckResult.evidence_analysis.sources_by_type
                                .academic
                            }
                          </div>
                          <div className="text-sm text-gray-600">Academic</div>
                        </div>
                        <div className="p-3 bg-purple-50 rounded-lg text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {
                              factCheckResult.evidence_analysis.sources_by_type
                                .news
                            }
                          </div>
                          <div className="text-sm text-gray-600">News</div>
                        </div>
                        <div className="p-3 bg-orange-50 rounded-lg text-center">
                          <div className="text-2xl font-bold text-orange-600">
                            {
                              factCheckResult.evidence_analysis.sources_by_type
                                .official
                            }
                          </div>
                          <div className="text-sm text-gray-600">Official</div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600">
                        Total sources analyzed:{" "}
                        {factCheckResult.evidence_analysis.total_sources} |
                        Average credibility:{" "}
                        {Math.round(
                          factCheckResult.evidence_analysis
                            .average_credibility * 100
                        )}
                        %
                      </p>
                    </div>

                    {/* Evidence Summary */}
                    <div className="border-t pt-4 mb-6">
                      <h3 className="text-lg font-semibold mb-3">
                        Evidence Summary
                      </h3>
                      <p className="text-gray-700 mb-4">
                        {factCheckResult.verdict_analysis.evidence_summary}
                      </p>
                    </div>

                    {/* Risk Factors */}
                    {factCheckResult.verdict_analysis.risk_factors.length >
                      0 && (
                      <div className="border-t pt-4 mb-6">
                        <h3 className="text-lg font-semibold mb-3 text-red-600">
                          Risk Factors
                        </h3>
                        <ul className="space-y-2">
                          {factCheckResult.verdict_analysis.risk_factors.map(
                            (factor: string, index: number) => (
                              <li key={index} className="flex items-start">
                                <span className="text-red-500 mr-2">•</span>
                                <span className="text-gray-700">{factor}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {/* Verification Suggestions */}
                    {factCheckResult.verdict_analysis.verification_suggestions
                      .length > 0 && (
                      <div className="border-t pt-4 mb-6">
                        <h3 className="text-lg font-semibold mb-3 text-blue-600">
                          Verification Suggestions
                        </h3>
                        <ul className="space-y-2">
                          {factCheckResult.verdict_analysis.verification_suggestions.map(
                            (suggestion: string, index: number) => (
                              <li key={index} className="flex items-start">
                                <span className="text-blue-500 mr-2">•</span>
                                <span className="text-gray-700">
                                  {suggestion}
                                </span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                    {/* Processing Info */}
                    <div className="border-t pt-4 text-xs text-gray-500">
                      <p>
                        Processing time:{" "}
                        {
                          factCheckResult.processing_metadata
                            .processing_time_seconds
                        }
                        s
                      </p>
                      <p>
                        Version: {factCheckResult.processing_metadata.version}
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <button
                onClick={resetState}
                className="mt-8 px-8 py-3 bg-gray-600 text-white font-bold rounded-lg shadow-md hover:bg-gray-800 transition-all duration-300"
              >
                Verify Another
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
