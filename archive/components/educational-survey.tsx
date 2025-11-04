"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "./ui/button";

const questions = [
  {
    question: "How much plastic enters the ocean each year?",
    options: ["1 million tons", "8 million tons", "500,000 tons"],
    correct: 1,
    explanation: "8 million tons of plastic waste enter our oceans annually, equivalent to dumping a garbage truck of plastic into the ocean every minute."
  },
  {
    question: "How long does plastic take to decompose in the ocean?",
    options: ["50 years", "200 years", "450+ years"],
    correct: 2,
    explanation: "Plastic takes 450+ years to decompose in the ocean. Some types of plastic may never fully break down, instead fragmenting into microplastics."
  },
  {
    question: "What is the Great Pacific Garbage Patch?",
    options: ["A floating island", "A plastic accumulation zone", "An ocean trench"],
    correct: 1,
    explanation: "The Great Pacific Garbage Patch is a massive accumulation zone of plastic debris spanning 1.6 million km², roughly twice the size of Texas."
  },
  {
    question: "What percentage of marine species are affected by plastic?",
    options: ["30%", "50%", "90%+"],
    correct: 2,
    explanation: "Over 90% of marine species have interacted with plastic pollution. Marine animals often mistake plastic for food, leading to starvation and death."
  },
  {
    question: "How can volunteer computing help ocean cleanup?",
    options: ["Track ocean currents", "Predict plastic drift", "Both"],
    correct: 2,
    explanation: "Volunteer computing helps run thousands of simulations to predict where plastic will drift, allowing cleanup crews to intercept it before it reaches sensitive habitats."
  }
];

export function EducationalSurvey() {
  const [isVisible, setIsVisible] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [score, setScore] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [isDismissed, setIsDismissed] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 800 && !isDismissed) {
        setIsVisible(true);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [isDismissed]);

  const handleAnswerSelect = (index: number) => {
    if (showExplanation) return;

    setSelectedAnswer(index);
    setShowExplanation(true);

    if (index === questions[currentQuestion].correct) {
      setScore(score + 1);
    }
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowExplanation(false);
    } else {
      setIsComplete(true);
    }
  };

  const handleClose = () => {
    setIsVisible(false);
    setIsDismissed(true);
  };

  const handleJoinNow = () => {
    const downloadSection = document.getElementById("download");
    if (downloadSection) {
      downloadSection.scrollIntoView({ behavior: "smooth" });
    }
    handleClose();
  };

  return (
    <AnimatePresence>
      {isVisible && !isDismissed && (
        <motion.div
          initial={{ opacity: 0, x: 400 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 400 }}
          transition={{ type: "spring", damping: 25, stiffness: 200 }}
          className="fixed top-24 right-6 z-50 w-[380px] max-w-[calc(100vw-48px)]"
        >
          <div className="bg-background border border-primary/30 rounded-lg shadow-2xl shadow-primary/20 overflow-hidden">
            <div className="bg-primary/10 border-b border-primary/30 px-6 py-4 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-lg">Ocean Plastic Quiz</h3>
                <p className="text-xs text-foreground/60 mt-1">
                  Test your knowledge about ocean pollution
                </p>
              </div>
              <button
                onClick={handleClose}
                className="text-foreground/60 hover:text-foreground transition-colors"
                aria-label="Close"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-6">
              {!isComplete ? (
                <>
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-foreground/60 mb-2">
                      <span>Question {currentQuestion + 1} of {questions.length}</span>
                      <span>Score: {score}/{questions.length}</span>
                    </div>
                    <div className="h-1 bg-foreground/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                      />
                    </div>
                  </div>

                  <p className="font-bold mb-4 text-sm leading-relaxed">
                    {questions[currentQuestion].question}
                  </p>

                  <div className="space-y-2 mb-4">
                    {questions[currentQuestion].options.map((option, index) => (
                      <button
                        key={index}
                        onClick={() => handleAnswerSelect(index)}
                        disabled={showExplanation}
                        className={`w-full text-left px-4 py-3 rounded-lg border transition-all text-sm ${
                          selectedAnswer === index
                            ? index === questions[currentQuestion].correct
                              ? "bg-green-500/20 border-green-500 text-green-400"
                              : "bg-red-500/20 border-red-500 text-red-400"
                            : showExplanation && index === questions[currentQuestion].correct
                            ? "bg-green-500/20 border-green-500 text-green-400"
                            : "bg-foreground/5 border-foreground/10 hover:border-primary/50 hover:bg-foreground/10"
                        }`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>

                  {showExplanation && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-primary/10 border border-primary/30 rounded-lg p-4 mb-4"
                    >
                      <p className="text-xs text-foreground/80 leading-relaxed">
                        {questions[currentQuestion].explanation}
                      </p>
                    </motion.div>
                  )}

                  {showExplanation && (
                    <Button
                      onClick={handleNext}
                      className="w-full"
                      size="sm"
                    >
                      {currentQuestion < questions.length - 1 ? "Next Question" : "See Results"}
                    </Button>
                  )}
                </>
              ) : (
                <div className="text-center py-4">
                  <div className="text-5xl mb-4">
                    {score === questions.length ? "🏆" : score >= 3 ? "🌊" : "🌍"}
                  </div>
                  <h4 className="font-bold text-xl mb-2">
                    You scored {score}/{questions.length}!
                  </h4>
                  <p className="text-sm text-foreground/70 mb-6 leading-relaxed">
                    {score === questions.length
                      ? "Perfect! You're an ocean conservation expert. Help us fight plastic pollution by contributing your computer's power."
                      : score >= 3
                      ? "Great job! Now help make a real impact by running ocean drift simulations."
                      : "Every bit helps! Join our volunteer network to fight ocean plastic pollution."}
                  </p>
                  <Button onClick={handleJoinNow} className="w-full">
                    Download App & Join
                  </Button>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
