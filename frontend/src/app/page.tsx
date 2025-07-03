"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import Image from "next/image";

// 다국어 텍스트 정의
const translations = {
  영어: {
    title: "Korea Visa Information",
    subtitle: "Search for the visa information you need",
    placeholder: "Start typing your question...",
    searchButton: "Search",
    examples: [
      "Student visa application process",
      "Tourist visa required documents",
      "Visa extension procedure",
      "Working visa eligibility",
      "Business visa requirements",
      "Visa application fees",
      "Processing time for visa",
      "Visa interview preparation",
    ],
    tagline: "Your Visa Navigation Assistant",
    searching: "Searching...",
  },
  한국어: {
    title: "한국 비자 정보",
    subtitle: "궁금한 비자 정보를 검색해보세요",
    placeholder: "질문을 입력해주세요...",
    searchButton: "검색",
    examples: [
      "학생 비자 신청 방법",
      "관광 비자 필요 서류",
      "비자 연장 절차",
      "취업 비자 자격 요건",
      "사업 비자 신청 조건",
      "비자 신청 수수료",
      "비자 처리 기간",
      "비자 면접 준비 사항",
    ],
    tagline: "비자 정보 안내 도우미",
    searching: "검색 중...",
  },
  중국어: {
    title: "韩国签证信息",
    subtitle: "搜索您需要的签证信息",
    placeholder: "请输入您的问题...",
    searchButton: "搜索",
    examples: [
      "学生签证申请流程",
      "旅游签证所需文件",
      "签证延期程序",
      "工作签证资格",
      "商务签证要求",
      "签证申请费用",
      "签证处理时间",
      "签证面试准备",
    ],
    tagline: "您的签证导航助手",
    searching: "搜索中...",
  },
  베트남어: {
    title: "Thông tin Visa Hàn Quốc",
    subtitle: "Tìm kiếm thông tin visa bạn cần",
    placeholder: "Bắt đầu nhập câu hỏi của bạn...",
    searchButton: "Tìm kiếm",
    examples: [
      "Quy trình xin visa du học",
      "Giấy tờ cần thiết cho visa du lịch",
      "Thủ tục gia hạn visa",
      "Điều kiện visa làm việc",
      "Yêu cầu visa kinh doanh",
      "Phí xin visa",
      "Thời gian xử lý visa",
      "Chuẩn bị phỏng vấn visa",
    ],
    tagline: "Trợ lý điều hướng Visa của bạn",
    searching: "Đang tìm kiếm...",
  },
  일본어: {
    title: "韓国ビザ情報",
    subtitle: "必要なビザ情報を検索してください",
    placeholder: "質問を入力してください...",
    searchButton: "検索",
    examples: [
      "学生ビザ申請手続き",
      "観光ビザ必要書類",
      "ビザ延長手続き",
      "就労ビザ資格",
      "商用ビザ要件",
      "ビザ申請料金",
      "ビザ処理期間",
      "ビザ面接準備",
    ],
    tagline: "あなたのビザナビゲーションアシスタント",
    searching: "検索中...",
  },
};

// 언어 옵션
const languageOptions = [
  { code: "en", name: "English", flag: "🇺🇸", lang: "영어" },
  { code: "ko", name: "한국어", flag: "🇰🇷", lang: "한국어" },
  { code: "zh", name: "中文", flag: "🇨🇳", lang: "중국어" },
  { code: "vi", name: "Tiếng Việt", flag: "🇻🇳", lang: "베트남어" },
  { code: "ja", name: "日本語", flag: "🇯🇵", lang: "일본어" },
];

export default function Home() {
  const [searchQuery, setSearchQuery] = useState("");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("영어");
  const router = useRouter();

  const currentTexts =
    translations[selectedLanguage as keyof typeof translations];

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsTransitioning(true);

    // 언어 정보도 함께 전달
    setTimeout(() => {
      router.push(
        `/chat?q=${encodeURIComponent(searchQuery)}&lang=${encodeURIComponent(
          selectedLanguage
        )}`
      );
    }, 800);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800 relative overflow-hidden">
      {/* 언어 선택 드롭다운 */}
      <div className="absolute top-6 right-6 z-20">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.4 }}
          className="relative"
        >
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="appearance-none bg-white/90 backdrop-blur-sm border-0 rounded-xl px-4 py-2 pr-10 text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-white/30 transition-all duration-200 cursor-pointer"
          >
            {languageOptions.map((lang) => (
              <option key={lang.code} value={lang.lang}>
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
            <svg
              className="w-4 h-4 text-gray-600"
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

      {/* 배경 패턴 */}
      {/* <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-20 w-32 h-32 bg-white rounded-full blur-xl"></div>
        <div className="absolute bottom-32 right-20 w-48 h-48 bg-white rounded-full blur-xl"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-white rounded-full blur-lg"></div>
      </div> */}

      <AnimatePresence>
        {!isTransitioning && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.6 }}
            className="relative z-10 flex flex-col lg:flex-row items-center justify-between min-h-screen px-8 lg:px-16 py-12 lg:py-0"
          >
            {/* 상단/왼쪽 섹션 - 브랜딩 (모바일에서는 상단) */}
            <div className="flex-1 lg:flex-none lg:w-1/3 flex items-center justify-center order-1 lg:order-1 mb-8 lg:mb-0 lg:mr-16 ml-8">
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="text-center lg:text-left"
              >
                <div className="relative ">
                  <h2 className="text-white text-6xl sm:text-7xl lg:text-7xl xl:text-8xl font-bold">
                    Na<span className="text-yellow-300">VISA</span>tion
                  </h2>
                  <div className="absolute -top-2 -right-2 lg:-top-3 lg:-right-3 w-6 h-6 lg:w-6 lg:h-6 bg-yellow-300 rounded-full" />
                </div>
                <p className="text-blue-100 text-lg lg:text-lg mt-2">
                  {currentTexts.tagline}
                </p>
              </motion.div>
            </div>

            {/* 하단/오른쪽 섹션 - 검색창 (모바일에서는 하단) */}
            <div className="flex-1 lg:flex-auto w-full max-w-xl lg:max-w-2xl order-2 lg:order-2 md:ml-16 lg:pr-8">
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.6 }}
              >
                <h1 className="hidden lg:block text-white text-3xl sm:text-4xl lg:text-4xl font-bold mb-2 text-center lg:text-left">
                  {currentTexts.title}
                </h1>
                <p className="hidden lg:block text-blue-100 text-lg lg:text-xl mb-6 text-center lg:text-left">
                  {currentTexts.subtitle}
                </p>

                <form onSubmit={handleSearch} className="space-y-4">
                  <div className="relative">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder={currentTexts.placeholder}
                      className="w-full px-6 py-4 text-black text-lg bg-white/90 backdrop-blur-sm border-0 rounded-2xl focus:outline-none focus:ring-4 focus:ring-white/30 transition-all duration-300 placeholder-gray-500 pr-20"
                    />
                    <motion.button
                      type="submit"
                      disabled={!searchQuery.trim()}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="absolute right-2 top-2 bottom-2 px-4 sm:px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-xl font-medium transition-colors duration-200 flex items-center gap-2 disabled:cursor-not-allowed"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                      <span className="hidden sm:inline">
                        {currentTexts.searchButton}
                      </span>
                    </motion.button>
                  </div>
                </form>

                {/* 슬라이딩 예시 텍스트 */}
                <div className="overflow-hidden">
                  <motion.div
                    animate={{ x: [-400, 0] }}
                    transition={{
                      duration: 20,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                    className="flex space-x-8 whitespace-nowrap"
                  >
                    {/* 첫 번째 세트 */}
                    {currentTexts.examples.map((example, index) => (
                      <span
                        key={`first-${index}`}
                        className="text-blue-100/70 text-sm px-4 py-2 rounded-full"
                      >
                        "{example}"
                      </span>
                    ))}
                    {/* 두 번째 세트 (무한 스크롤을 위한 복사본) */}
                    {currentTexts.examples.map((example, index) => (
                      <span
                        key={`second-${index}`}
                        className="text-blue-100/70 text-sm px-4 py-2 rounded-full"
                      >
                        "{example}"
                      </span>
                    ))}
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 전환 애니메이션 */}
      <AnimatePresence>
        {isTransitioning && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-blue-600"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 border-4 border-white border-t-transparent rounded-full"
            />
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-white text-xl ml-4"
            >
              {currentTexts.searching}
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
