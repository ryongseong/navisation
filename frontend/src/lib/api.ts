export async function fetchAnswer(
  question: string,
  language: string = "en"
): Promise<{ question: string; answer: string }> {
  const res = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/chat-request?req=${encodeURIComponent(
      question
    )}&lang=${encodeURIComponent(language)}`
  );
  if (!res.ok) throw new Error("Failed to fetch answer");
  return res.json(); // { question, answer }
}
