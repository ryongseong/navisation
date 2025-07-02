export async function fetchAnswer(question: string) : Promise<{ question: string; answer: string }> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat-request?req=${encodeURIComponent(question)}`)
  if (!res.ok) throw new Error('Failed to fetch answer')
  return res.json() // { question, answer }
}
