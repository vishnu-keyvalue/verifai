/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useMemo, useEffect, useRef } from "react";
import {
  Upload,
  Link,
  FileText,
  CheckCircle,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Shield,
  Search,
  Clock,
  TrendingUp,
  Users,
  Globe,
  Award,
  RefreshCw,
  ExternalLink,
  Info,
} from "lucide-react";

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

// Type for the Text Fact-Check result (simpler format)
interface TextFactCheckResult {
  claim: string;
  verdict: string;
  confidence_score: number;
  explanation: string;
  evidence_summary: string;
  risk_factors: string[];
  verification_suggestions: string[];
  processing_metadata: {
    timestamp: string;
    version: string;
  };
}

// --- SVG Icon Components ---
// It's good practice to define icons as reusable components

const LogoIcon = () => (
  <div className="relative">
    <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-800 rounded-xl flex items-center justify-center shadow-lg">
      <Shield className="w-6 h-6 text-white" />
    </div>
    <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
      <CheckCircle className="w-2.5 h-2.5 text-white" />
    </div>
  </div>
);

// Enhanced Verdict Icon Component
const VerdictIcon = ({ verdict }: { verdict: string }) => {
  const getVerdictStyle = (verdict: string) => {
    if (verdict.includes("True") || verdict.includes("Authentic")) {
      return {
        bgColor: "bg-green-100",
        iconColor: "text-green-600",
        icon: CheckCircle,
        ringColor: "ring-green-200",
      };
    }
    if (verdict.includes("False") || verdict.includes("AI Detected")) {
      return {
        bgColor: "bg-red-100",
        iconColor: "text-red-600",
        icon: XCircle,
        ringColor: "ring-red-200",
      };
    }
    if (
      verdict.includes("Uncertain") ||
      verdict.includes("Inconclusive") ||
      verdict.includes("Misleading") ||
      verdict.includes("Contested") ||
      verdict.includes("Unproven") ||
      verdict.includes("Insufficient")
    ) {
      return {
        bgColor: "bg-yellow-100",
        iconColor: "text-yellow-600",
        icon: AlertTriangle,
        ringColor: "ring-yellow-200",
      };
    }
    return {
      bgColor: "bg-gray-100",
      iconColor: "text-gray-600",
      icon: HelpCircle,
      ringColor: "ring-gray-200",
    };
  };

  const style = getVerdictStyle(verdict);
  const IconComponent = style.icon;

  return (
    <div
      className={`w-16 h-16 ${style.bgColor} rounded-full flex items-center justify-center ring-4 ${style.ringColor} shadow-lg`}
    >
      <IconComponent className={`w-8 h-8 ${style.iconColor}`} />
    </div>
  );
};

// Progress Bar Component
const ProgressBar = ({ progress }: { progress: number }) => (
  <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
    <div
      className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
      style={{ width: `${progress}%` }}
    />
  </div>
);

// Stats Card Component
const StatsCard = ({
  icon: Icon,
  label,
  value,
  color = "blue",
}: {
  icon: any;
  label: string;
  value: string | number;
  color?: string;
}) => {
  const colorClasses = {
    blue: "bg-blue-50 text-blue-600 border-blue-200",
    green: "bg-green-50 text-green-600 border-green-200",
    purple: "bg-purple-50 text-purple-600 border-purple-200",
    orange: "bg-orange-50 text-orange-600 border-orange-200",
    red: "bg-red-50 text-red-600 border-red-200",
    yellow: "bg-yellow-50 text-yellow-600 border-yellow-200",
  };

  return (
    <div
      className={`p-4 rounded-xl border-2 ${
        colorClasses[color as keyof typeof colorClasses]
      } transition-all duration-200 hover:shadow-md`}
    >
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-5 h-5" />
        <span className="text-2xl font-bold">{value}</span>
      </div>
      <div className="text-sm font-medium opacity-80">{label}</div>
    </div>
  );
};

// Confidence Meter Component
const ConfidenceMeter = ({
  confidence,
  label,
}: {
  confidence: number;
  label: string;
}) => {
  const getColor = (conf: number) => {
    if (conf >= 80) return "text-green-600 bg-green-100";
    if (conf >= 60) return "text-yellow-600 bg-yellow-100";
    return "text-red-600 bg-red-100";
  };

  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <div className="flex items-center gap-2">
        <div className="w-20 bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${
              confidence >= 80
                ? "bg-green-500"
                : confidence >= 60
                ? "bg-yellow-500"
                : "bg-red-500"
            }`}
            style={{ width: `${confidence}%` }}
          />
        </div>
        <span
          className={`text-sm font-bold px-2 py-1 rounded ${getColor(
            confidence
          )}`}
        >
          {confidence}%
        </span>
      </div>
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
  const [textFactCheckResult, setTextFactCheckResult] =
    useState<TextFactCheckResult | null>(null);

  // States for the loading animation
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const loadingMessages = [
    { text: "Initializing analysis...", progress: 10 },
    { text: "Scanning for digital artifacts...", progress: 35 },
    { text: "Cross-referencing sources...", progress: 65 },
    { text: "Generating comprehensive report...", progress: 90 },
    { text: "Finalizing results...", progress: 100 },
  ];

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Effect to cycle through loading messages with progress
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (verificationState === "loading") {
      interval = setInterval(() => {
        setLoadingMessageIndex((prevIndex) => {
          const nextIndex = (prevIndex + 1) % loadingMessages.length;
          setLoadingProgress(loadingMessages[nextIndex].progress);
          return nextIndex;
        });
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [loadingMessages, loadingMessages.length, verificationState]);

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
    setTextFactCheckResult(null);
    setLoadingProgress(0);
    setLoadingMessageIndex(0);
  };

  const handleVerify = async () => {
    if (!hasContent) return;
    setVerificationState("loading");
    setLoadingMessageIndex(0);
    setLoadingProgress(10);

    // Clear previous results before starting a new analysis
    setAnalysisDetails(null);
    setFactCheckResult(null);
    setTextFactCheckResult(null);

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
    } else if (activeTab === "text" && pastedText) {
      endpoint = "http://127.0.0.1:5001/api/v1/fact-check/text";
      requestBody = JSON.stringify({ text: pastedText });
      headers["Content-Type"] = "application/json";
    } else {
      alert("No content to verify.");
      setVerificationState("idle");
      return;
    }

    try {
      console.log("Making API call to:", endpoint);
      console.log("Request body:", requestBody);
      console.log("Headers:", headers);

      const response = await fetch(endpoint, {
        method: "POST",
        body: requestBody,
        headers: headers,
      });

      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error response:", errorData);
        throw new Error(
          errorData.error ||
            `Network response was not ok: ${response.statusText}`
        );
      }

      const data = await response.json();
      console.log("Response data:", data);

      // Check the structure of the response to determine how to set the state
      if (data.verdict_analysis) {
        // This is a fact-check response with full analysis
        setFactCheckResult(data as FactCheckResult);
      } else if (data.claim && data.verdict && !data.verdict_analysis) {
        // This is a text fact-check response
        setTextFactCheckResult(data as TextFactCheckResult);
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

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(false);
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
      return factCheckResult.verdict_analysis?.verdict || "Analysis Complete";
    }
    if (textFactCheckResult) {
      return textFactCheckResult.verdict;
    }
    return "Analysis Complete";
  };

  const getTabIcon = (tab: ActiveTab) => {
    switch (tab) {
      case "upload":
        return Upload;
      case "link":
        return Link;
      case "text":
        return FileText;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 max-w-6xl">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <LogoIcon />
              <div>
                <span className="text-2xl font-bold text-gray-900 font-poppins">
                  VerifAI
                </span>
                <p className="text-sm text-gray-600">AI-Powered Verification</p>
              </div>
            </div>
            <nav className="hidden md:flex gap-6 text-gray-600">
              <a
                href="#"
                className="hover:text-blue-600 transition-colors flex items-center gap-2"
              >
                <Info className="w-4 h-4" />
                About
              </a>
              <a
                href="#"
                className="hover:text-blue-600 transition-colors flex items-center gap-2"
              >
                <HelpCircle className="w-4 h-4" />
                How it Works
              </a>
            </nav>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <main>
          {verificationState === "idle" && (
            <div className="animate-fade-in">
              {/* Hero Section */}
              <div className="text-center mb-12">
                <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent font-poppins">
                  AI-Powered Content Verification
                </h1>
                <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
                  Analyze images, links, and text to detect AI generation and
                  verify factual claims with advanced machine learning
                  algorithms.
                </p>

                {/* Feature Pills */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                  <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200">
                    <Shield className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium">AI Detection</span>
                  </div>
                  <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200">
                    <Search className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium">Fact Checking</span>
                  </div>
                  <div className="flex items-center gap-2 bg-white/60 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200">
                    <TrendingUp className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-medium">Source Analysis</span>
                  </div>
                </div>
              </div>

              {/* Main Input Card */}
              <div className="bg-white/80 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-gray-200 max-w-3xl mx-auto">
                {/* Tab Navigation */}
                <div className="flex bg-gray-100 p-1 rounded-xl mb-6">
                  {(["upload", "link", "text"] as ActiveTab[]).map((tab) => {
                    const Icon = getTabIcon(tab);
                    return (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 font-semibold capitalize transition-all duration-300 rounded-lg ${
                          activeTab === tab
                            ? "bg-white text-blue-600 shadow-md"
                            : "text-gray-500 hover:text-gray-700"
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        {tab === "upload"
                          ? "Upload File"
                          : tab === "link"
                          ? "Paste Link"
                          : "Paste Text"}
                      </button>
                    );
                  })}
                </div>

                {/* Input Areas */}
                {activeTab === "upload" && (
                  <div>
                    <label
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      className={`flex flex-col items-center justify-center w-full h-48 px-4 transition-all duration-300 bg-gray-50 border-2 border-dashed rounded-xl cursor-pointer hover:bg-gray-100 ${
                        isDragOver
                          ? "border-blue-500 bg-blue-50"
                          : "border-gray-300"
                      } ${uploadedFile ? "border-green-500 bg-green-50" : ""}`}
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        {uploadedFile ? (
                          <>
                            <CheckCircle className="w-12 h-12 mb-4 text-green-500" />
                            <p className="mb-2 text-lg font-semibold text-green-700">
                              File Selected
                            </p>
                            <p className="text-sm text-green-600">
                              {uploadedFile.name}
                            </p>
                          </>
                        ) : (
                          <>
                            <Upload
                              className={`w-12 h-12 mb-4 ${
                                isDragOver ? "text-blue-500" : "text-gray-400"
                              }`}
                            />
                            <p className="mb-2 text-lg font-semibold text-gray-700">
                              Drop files here or click to browse
                            </p>
                            <p className="text-sm text-gray-500">
                              Supports JPG, PNG (Max 10MB)
                            </p>
                          </>
                        )}
                      </div>
                      <input
                        type="file"
                        className="hidden"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept="image/*"
                      />
                    </label>
                  </div>
                )}

                {activeTab === "link" && (
                  <div className="space-y-4">
                    <div className="relative">
                      <Link className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                      <input
                        type="url"
                        value={linkUrl}
                        onChange={(e) => setLinkUrl(e.target.value)}
                        placeholder="https://example.com/article-or-news-link"
                        className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 text-lg"
                      />
                    </div>
                    <p className="text-sm text-gray-500 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      We'll analyze the content and verify claims from news
                      articles, blog posts, and web pages
                    </p>
                  </div>
                )}

                {activeTab === "text" && (
                  <div className="space-y-4">
                    <div className="relative">
                      <FileText className="absolute left-3 top-4 text-gray-400 w-5 h-5" />
                      <textarea
                        value={pastedText}
                        onChange={(e) => setPastedText(e.target.value)}
                        placeholder="Paste text from an article, social media post, or any claim you want to verify..."
                        className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl h-32 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 text-lg resize-none"
                      />
                    </div>
                    <div className="flex justify-between items-center text-sm text-gray-500">
                      <div className="flex items-center gap-2">
                        <Info className="w-4 h-4" />
                        Minimum 10 characters required
                      </div>
                      <div>{pastedText.length} characters</div>
                    </div>
                  </div>
                )}

                {/* Verify Button */}
                <button
                  onClick={handleVerify}
                  disabled={!hasContent}
                  className="w-full mt-8 px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold rounded-xl shadow-lg hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] disabled:hover:scale-100 text-lg flex items-center justify-center gap-2"
                >
                  <Shield className="w-5 h-5" />
                  {hasContent ? "Verify Content" : "Select Content to Verify"}
                </button>
              </div>
            </div>
          )}

          {/* Enhanced Loading State */}
          {verificationState === "loading" && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] animate-fade-in">
              <div className="bg-white/80 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-gray-200 max-w-md w-full text-center">
                <div className="relative mb-6">
                  <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center animate-pulse">
                    <Shield className="w-10 h-10 text-white" />
                  </div>
                  <div className="absolute inset-0 w-20 h-20 mx-auto border-4 border-blue-200 rounded-full animate-spin border-t-blue-500"></div>
                </div>

                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  Analyzing Content
                </h3>

                <p className="text-gray-600 mb-6 transition-opacity duration-500">
                  {loadingMessages[loadingMessageIndex].text}
                </p>

                <ProgressBar progress={loadingProgress} />

                <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
                  <Clock className="w-4 h-4" />
                  This may take a few moments
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Results Display */}
          {verificationState === "complete" && (
            <div className="animate-fade-in space-y-6">
              {/* Main Result Card */}
              <div className="bg-white/80 backdrop-blur-md rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
                {/* Header */}
                <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-6 border-b border-gray-200">
                  <div className="flex items-center gap-6">
                    <VerdictIcon verdict={renderVerdictTitle()} />
                    <div className="flex-1">
                      <h2 className="text-sm text-gray-500 uppercase tracking-wide font-medium mb-1">
                        Verification Result
                      </h2>
                      <p className="text-3xl font-bold text-gray-900 font-poppins mb-2">
                        {renderVerdictTitle()}
                      </p>
                      {(analysisDetails || factCheckResult) && (
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <div className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {factCheckResult?.processing_metadata
                              ?.processing_time_seconds
                              ? `${factCheckResult.processing_metadata.processing_time_seconds}s`
                              : "< 1s"}
                          </div>
                          <div className="flex items-center gap-1">
                            <TrendingUp className="w-4 h-4" />
                            {analysisDetails
                              ? `${analysisDetails.confidence}% confidence`
                              : factCheckResult?.verdict_analysis
                                  ?.confidence_score
                              ? `${Math.round(
                                  factCheckResult.verdict_analysis
                                    .confidence_score * 100
                                )}% confidence`
                              : "Analysis complete"}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Content */}
                <div className="p-6">
                  {/* AI Detection Results */}
                  {analysisDetails && (
                    <div className="space-y-6">
                      <div className="prose max-w-none">
                        <p className="text-gray-700 text-lg leading-relaxed">
                          Our advanced AI model analyzed the image for digital
                          artifacts, inconsistencies, and patterns commonly
                          found in AI-generated content.
                        </p>
                      </div>

                      <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-gray-50 p-6 rounded-xl">
                          <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                            <Search className="w-5 h-5 text-blue-600" />
                            Authenticity Analysis
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {analysisDetails.reasons[0].text}
                          </p>
                        </div>

                        <div className="bg-gray-50 p-6 rounded-xl">
                          <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                            <Award className="w-5 h-5 text-purple-600" />
                            Model Prediction
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            Identified as{" "}
                            <span className="font-bold text-purple-600">
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

                  {/* Fact Check Results */}
                  {factCheckResult && (
                    <div className="space-y-8">
                      {/* Claim and Verdict */}
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200">
                        <h3 className="font-semibold text-lg mb-3 text-blue-800">
                          Primary Claim
                        </h3>
                        <blockquote className="text-gray-700 italic text-lg leading-relaxed border-l-4 border-blue-400 pl-4">
                          "
                          {factCheckResult.verdict_analysis?.primary_claim ||
                            "Analysis completed"}
                          "
                        </blockquote>
                      </div>

                      {/* Confidence Metrics */}
                      <div className="space-y-4">
                        <h3 className="font-semibold text-lg flex items-center gap-2">
                          <TrendingUp className="w-5 h-5 text-green-600" />
                          Confidence Metrics
                        </h3>
                        <div className="grid gap-3">
                          <ConfidenceMeter
                            confidence={Math.round(
                              (factCheckResult.verdict_analysis
                                ?.confidence_score || 0) * 100
                            )}
                            label="Overall Confidence"
                          />
                          {factCheckResult.verdict_analysis
                            ?.credibility_breakdown && (
                            <>
                              <ConfidenceMeter
                                confidence={Math.round(
                                  factCheckResult.verdict_analysis
                                    .credibility_breakdown
                                    .evidence_consistency * 100
                                )}
                                label="Evidence Consistency"
                              />
                              <ConfidenceMeter
                                confidence={Math.round(
                                  factCheckResult.verdict_analysis
                                    .credibility_breakdown
                                    .source_credibility_average * 100
                                )}
                                label="Source Credibility"
                              />
                              <ConfidenceMeter
                                confidence={Math.round(
                                  factCheckResult.verdict_analysis
                                    .credibility_breakdown.source_diversity *
                                    100
                                )}
                                label="Source Diversity"
                              />
                            </>
                          )}
                        </div>
                      </div>

                      {/* Source Analysis */}
                      {factCheckResult.evidence_analysis && (
                        <div className="space-y-4">
                          <h3 className="font-semibold text-lg flex items-center gap-2">
                            <Users className="w-5 h-5 text-purple-600" />
                            Source Analysis
                          </h3>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <StatsCard
                              icon={Shield}
                              label="Fact Checkers"
                              value={
                                factCheckResult.evidence_analysis
                                  .sources_by_type?.fact_check || 0
                              }
                              color="blue"
                            />
                            <StatsCard
                              icon={Award}
                              label="Academic"
                              value={
                                factCheckResult.evidence_analysis
                                  .sources_by_type?.academic || 0
                              }
                              color="green"
                            />
                            <StatsCard
                              icon={Globe}
                              label="News Sources"
                              value={
                                factCheckResult.evidence_analysis
                                  .sources_by_type?.news || 0
                              }
                              color="purple"
                            />
                            <StatsCard
                              icon={CheckCircle}
                              label="Official"
                              value={
                                factCheckResult.evidence_analysis
                                  .sources_by_type?.official || 0
                              }
                              color="orange"
                            />
                          </div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <div className="flex justify-between items-center text-sm">
                              <span className="text-gray-600">
                                Total sources analyzed:
                              </span>
                              <span className="font-bold">
                                {
                                  factCheckResult.evidence_analysis
                                    .total_sources
                                }
                              </span>
                            </div>
                            <div className="flex justify-between items-center text-sm mt-1">
                              <span className="text-gray-600">
                                Average credibility:
                              </span>
                              <span className="font-bold">
                                {Math.round(
                                  factCheckResult.evidence_analysis
                                    .average_credibility * 100
                                )}
                                %
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Evidence Summary */}
                      {factCheckResult.verdict_analysis?.evidence_summary && (
                        <div className="bg-gray-50 p-6 rounded-xl">
                          <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                            <Info className="w-5 h-5 text-blue-600" />
                            Evidence Summary
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {factCheckResult.verdict_analysis.evidence_summary}
                          </p>
                        </div>
                      )}

                      {/* Risk Factors */}
                      {factCheckResult.verdict_analysis?.risk_factors &&
                        factCheckResult.verdict_analysis.risk_factors.length >
                          0 && (
                          <div className="bg-red-50 border border-red-200 p-6 rounded-xl">
                            <h3 className="font-semibold text-lg mb-3 text-red-800 flex items-center gap-2">
                              <AlertTriangle className="w-5 h-5" />
                              Risk Factors
                            </h3>
                            <ul className="space-y-2">
                              {factCheckResult.verdict_analysis.risk_factors.map(
                                (factor: string, index: number) => (
                                  <li
                                    key={index}
                                    className="flex items-start gap-2"
                                  >
                                    <XCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                                    <span className="text-red-700">
                                      {factor}
                                    </span>
                                  </li>
                                )
                              )}
                            </ul>
                          </div>
                        )}

                      {/* Verification Suggestions */}
                      {factCheckResult.verdict_analysis
                        ?.verification_suggestions &&
                        factCheckResult.verdict_analysis
                          .verification_suggestions.length > 0 && (
                          <div className="bg-blue-50 border border-blue-200 p-6 rounded-xl">
                            <h3 className="font-semibold text-lg mb-3 text-blue-800 flex items-center gap-2">
                              <CheckCircle className="w-5 h-5" />
                              Verification Suggestions
                            </h3>
                            <ul className="space-y-2">
                              {factCheckResult.verdict_analysis.verification_suggestions.map(
                                (suggestion: string, index: number) => (
                                  <li
                                    key={index}
                                    className="flex items-start gap-2"
                                  >
                                    <CheckCircle className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                                    <span className="text-blue-700">
                                      {suggestion}
                                    </span>
                                  </li>
                                )
                              )}
                            </ul>
                          </div>
                        )}

                      {/* Detailed Explanation */}
                      {factCheckResult.verdict_analysis?.explanation && (
                        <div className="bg-gray-50 p-6 rounded-xl">
                          <h3 className="font-semibold text-lg mb-3">
                            Detailed Analysis
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {factCheckResult.verdict_analysis.explanation}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-4 justify-center">
                {factCheckResult?.source_content?.url && (
                  <a
                    href={factCheckResult.source_content.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-xl shadow-md hover:shadow-lg hover:bg-blue-700 transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <ExternalLink className="w-4 h-4" />
                    View Original Source
                  </a>
                )}

                {/* Text Fact Check Result Card */}
                {textFactCheckResult && (
                  <div>
                    <div className="mb-6">
                      <p className="text-gray-600 mb-2">
                        Claim:{" "}
                        <span className="font-semibold italic">
                          "{textFactCheckResult.claim}"
                        </span>
                      </p>
                      <p className="text-lg mb-4">
                        <span className="font-bold">Verdict:</span>{" "}
                        {textFactCheckResult.verdict}
                      </p>
                      <p className="text-lg mb-4">
                        <span className="font-bold">Confidence:</span>{" "}
                        {Math.round(textFactCheckResult.confidence_score * 100)}
                        %
                      </p>
                      <p className="text-gray-700 mb-4">
                        {textFactCheckResult.explanation}
                      </p>
                    </div>

                    {/* Evidence Summary */}
                    <div className="border-t pt-4 mb-6">
                      <h3 className="text-lg font-semibold mb-3">
                        Evidence Summary
                      </h3>
                      <p className="text-gray-700 mb-4">
                        {textFactCheckResult.evidence_summary}
                      </p>
                    </div>

                    {/* Risk Factors */}
                    {textFactCheckResult.risk_factors.length > 0 && (
                      <div className="border-t pt-4 mb-6">
                        <h3 className="text-lg font-semibold mb-3 text-red-600">
                          Risk Factors
                        </h3>
                        <ul className="space-y-2">
                          {textFactCheckResult.risk_factors.map(
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
                    {textFactCheckResult.verification_suggestions.length >
                      0 && (
                      <div className="border-t pt-4 mb-6">
                        <h3 className="text-lg font-semibold mb-3 text-blue-600">
                          Verification Suggestions
                        </h3>
                        <ul className="space-y-2">
                          {textFactCheckResult.verification_suggestions.map(
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
                        Version:{" "}
                        {textFactCheckResult.processing_metadata.version}
                      </p>
                      <p>
                        Timestamp:{" "}
                        {textFactCheckResult.processing_metadata.timestamp}
                      </p>
                    </div>
                  </div>
                )}
                <button
                  onClick={resetState}
                  className="px-8 py-3 bg-white text-gray-700 font-semibold rounded-xl shadow-md hover:shadow-lg border border-gray-200 transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Verify Another
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
