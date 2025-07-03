// app/chat/ChatClient.tsx
'use client'
import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { fetchAnswer } from "@/lib/api";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";


const translations = {
  영어: {
    initialMessage: "Hello! How can I help you?",
    placeholder: "Type your message... (Shift+Enter for line break)",
    sending: "Sending...",
    languageLabel: "Language:",
  },
  한국어: {
    initialMessage: "안녕하세요! 무엇을 도와드릴까요?",
    placeholder: "메시지를 입력하세요... (Shift+Enter로 줄바꿈)",
    sending: "전송 중...",
    languageLabel: "언어:",
  },
  중국어: {
    initialMessage: "您好！我可以为您做些什么？",
    placeholder: "输入您的消息... (Shift+Enter换行)",
    sending: "发送中...",
    languageLabel: "语言:",
  },
  베트남어: {
    initialMessage: "Xin chào! Tôi có thể giúp gì cho bạn?",
    placeholder: "Nhập tin nhắn của bạn... (Shift+Enter để xuống dòng)",
    sending: "Đang gửi...",
    languageLabel: "Ngôn ngữ:",
  },
  일본어: {
    initialMessage: "こんにちは！何をお手伝いできますか？",
    placeholder: "メッセージを入力してください... (Shift+Enterで改行)",
    sending: "送信中...",
    languageLabel: "言語:",
  },
};

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("영어");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const searchParams = useSearchParams();
  const hasProcessedQuery = useRef(false);

  const languageOptions = [
    { code: "en", name: "English", flag: "🇺🇸", lang: "영어" },
    { code: "ko", name: "한국어", flag: "🇰🇷", lang: "한국어" },
    { code: "zh", name: "中文", flag: "🇨🇳", lang: "중국어" },
    { code: "vi", name: "Tiếng Việt", flag: "🇻🇳", lang: "베트남어" },
    { code: "ja", name: "日本語", flag: "🇯🇵", lang: "일본어" },
  ];

  // 선택된 언어에 따른 텍스트
  const currentTexts =
    translations[selectedLanguage as keyof typeof translations];

  // 초기 메시지를 언어에 따라 설정
  const [messages, setMessages] = useState([
    { role: "assistant", content: currentTexts.initialMessage },
  ]);

  // 언어가 변경될 때 초기 메시지도 업데이트
  useEffect(() => {
    if (messages.length === 1 && messages[0].role === "assistant") {
      setMessages([
        { role: "assistant", content: currentTexts.initialMessage },
      ]);
    }
  }, [selectedLanguage, currentTexts.initialMessage]);

  // 메시지 변경 시 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // URL에서 검색어와 언어 정보 가져와서 자동 실행 (한 번만)
  useEffect(() => {
    const query = searchParams?.get("q");
    const lang = searchParams?.get("lang") || "영어";

    // 언어 설정
    setSelectedLanguage(lang);

    if (query && !hasProcessedQuery.current) {
      hasProcessedQuery.current = true;
      console.log("Processing initial query:", query, "Language:", lang);

      // 언어에 맞는 초기 메시지로 설정
      const langTexts =
        translations[lang as keyof typeof translations] || translations["영어"];
      setMessages([
        { role: "assistant", content: langTexts.initialMessage },
        { role: "user", content: query },
      ]);

      // 언어 정보와 함께 질문 실행
      mutation.mutate({ question: query, language: lang });
    }
  }, [searchParams]);

  // React Query mutation: 질문을 서버에 보내고 답변 받기
  const mutation = useMutation({
    mutationFn: ({
      question,
      language,
    }: {
      question: string;
      language?: string;
    }) => {
      console.log(
        "Sending question to server:",
        question,
        "Language:",
        language
      );
      // 언어 정보도 함께 전송
      return fetchAnswer(question, language);
    },
    onSuccess: (data) => {
      console.log("Received answer:", data.answer);
      // 서버에서 답변 받으면 assistant 메시지만 추가
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    },
    onError: (error) => {
      console.log(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "죄송합니다. 답변을 가져오는 중 오류가 발생했습니다.",
        },
      ]);
    },
  });

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // 유저 메시지를 먼저 추가
    setMessages((prev) => [...prev, { role: "user", content: input }]);
    // 언어 정보와 함께 질문 보내기
    mutation.mutate({ question: input, language: selectedLanguage });
    setInput("");
  };

  return (
    <div className="flex flex-col h-screen bg-[#f7f7f8] dark:bg-[#343541]">
      <header className="w-full px-6 py-4 border-b border-[#ececf1] dark:border-[#202123] bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800 flex items-center justify-between">
        <Link
          href="/"
          className="text-lg font-semibold text-white select-none group"
        >
          <motion.span
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
            className="inline-block"
          >
            Na<span className="text-yellow-300">VISA</span>tion
          </motion.span>
        </Link>

        <div className="flex items-center gap-4">
          {/* 언어 표시 */}
          {/* <div className="text-sm text-gray-600 dark:text-gray-300">
            {currentTexts.languageLabel} {selectedLanguage}
          </div> */}

          {/* 언어 선택 드롭다운 */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="relative"
          >
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="appearance-none bg-white dark:bg-[#40414f] border border-[#ececf1] dark:border-[#444654] rounded-lg px-3 py-1.5 pr-8 text-sm text-gray-800 dark:text-white font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 cursor-pointer"
            >
              {languageOptions.map((lang) => (
                <option key={lang.code} value={lang.lang}>
                  {lang.flag} {lang.name}
                </option>
              ))}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
              <svg
                className="w-3 h-3 text-gray-600 dark:text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </div>
          </motion.div>
        </div>
      </header>
      <main className="flex-1 overflow-y-auto w-full max-w-2xl mx-auto px-2 sm:px-0 py-6 custom-scrollbar">
        <div className="flex flex-col gap-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`px-4 py-3 max-w-[80%] text-base whitespace-pre-line rounded-2xl shadow-sm border ${
                  msg.role === "user"
                    ? "bg-[#007aff] text-white border-[#007aff] rounded-br-md"
                    : "bg-[#ececf1] dark:bg-[#444654] text-gray-900 dark:text-[#ececf1] border-[#ececf1] dark:border-[#444654] rounded-bl-md"
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {/* 로딩 UI */}
          {mutation.isPending && (
            <div className="flex justify-start">
              <div className="px-4 py-3 max-w-[80%] bg-[#ececf1] dark:bg-[#444654] border border-[#ececf1] dark:border-[#444654] rounded-2xl rounded-bl-md shadow-sm">
                <div className="flex items-center gap-2">
                  {/* 타이핑 애니메이션 */}
                  <div className="flex space-x-1">
                    <motion.div
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: 0 }}
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full"
                    />
                    <motion.div
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{
                        duration: 1.2,
                        repeat: Infinity,
                        delay: 0.2,
                      }}
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full"
                    />
                    <motion.div
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{
                        duration: 1.2,
                        repeat: Infinity,
                        delay: 0.4,
                      }}
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>
      <form
        onSubmit={handleSend}
        className="w-full max-w-2xl mx-auto px-2 sm:px-0 pb-6 flex gap-2 sticky bottom-0 bg-gradient-to-t from-[#f7f7f8] via-[#f7f7f8]/80 to-transparent dark:from-[#343541] dark:via-[#343541]/80"
        style={{ zIndex: 10 }}
      >
        <input
          type="text"
          className="flex-1 rounded-xl border border-[#ececf1] dark:border-[#444654] px-4 py-3 text-base bg-white dark:bg-[#40414f] text-gray-900 dark:text-[#ececf1] placeholder-gray-400 dark:placeholder-[#8e8ea0] focus:outline-none focus:ring-2 focus:ring-[#007aff] dark:focus:ring-[#007aff] transition shadow"
          placeholder={currentTexts.placeholder}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend(e as unknown as React.FormEvent<HTMLFormElement>);
            }
          }}
          disabled={mutation.isPending}
        />
        <button
          type="submit"
          className="bg-[#007aff] hover:bg-[#005bb5] text-white font-semibold px-6 py-3 rounded-xl transition shadow disabled:opacity-50 border border-[#007aff]"
          disabled={!input.trim() || mutation.isPending}
        >
          {mutation.isPending ? (
            currentTexts.sending
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M5 12h14M12 5l7 7-7 7"
              />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
}
