export async function fetchAnswer(
  question: string,
  language?: string,
  sessionId?: string
) {
  const params = new URLSearchParams({
    req: question,
    lang: language || "한국어",
    session_id: sessionId || "default",
  });

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/chat-request?${params.toString()}`
  );

  if (!res.ok) throw new Error("Failed to fetch answer");

  const data = await res.json();
  return data; // { question, answer }
}
