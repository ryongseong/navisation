'use client'

import {useState, useRef, useEffect} from 'react'
import {useMutation} from '@tanstack/react-query'
import {fetchAnswer} from '@/lib/api'

export default function ChatPage() {
    const [messages, setMessages] = useState([
        {role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?'},
    ])
    const [input, setInput] = useState('')
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'})
    }, [messages])

    // React Query mutation: 질문을 서버에 보내고 답변 받기
    const mutation = useMutation({
        mutationFn: (question: string) => fetchAnswer(question),
        onSuccess: (data) => {
            // 서버에서 답변 받으면 assistant 메시지만 추가
            setMessages((prev) => [
                ...prev,
                {role: 'assistant', content: data.answer},
            ])
        },
        onError: (error) => {
            console.log(error)
            setMessages((prev) => [
                ...prev,
                {role: 'assistant', content: '죄송합니다. 답변을 가져오는 중 오류가 발생했습니다.'},
            ])
        },
    })

    const handleSend = (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        // 유저 메시지를 먼저 추가
        setMessages((prev) => [
            ...prev,
            {role: 'user', content: input},
        ])
        // 질문 보내기
        mutation.mutate(input)
        setInput('')
    }

    return (
        <div className="flex flex-col h-screen bg-[#f7f7f8] dark:bg-[#343541]">
            <header
                className="w-full px-6 py-4 border-b border-[#ececf1] dark:border-[#202123] bg-white dark:bg-[#343541] flex items-center">
                <span className="text-lg font-semibold text-gray-900 dark:text-white select-none">naVISAtion</span>
            </header>
            <main className="flex-1 overflow-y-auto w-full max-w-2xl mx-auto px-2 sm:px-0 py-6">
                <div className="flex flex-col gap-4">
                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`px-4 py-3 max-w-[80%] text-base whitespace-pre-line rounded-2xl shadow-sm border ${
                                    msg.role === 'user'
                                        ? 'bg-[#007aff] text-white border-[#007aff] rounded-br-md'
                                        : 'bg-[#ececf1] dark:bg-[#444654] text-gray-900 dark:text-[#ececf1] border-[#ececf1] dark:border-[#444654] rounded-bl-md'
                                }`}
                            >
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef}/>
                </div>
            </main>
            <form
                onSubmit={handleSend}
                className="w-full max-w-2xl mx-auto px-2 sm:px-0 pb-6 flex gap-2 sticky bottom-0 bg-gradient-to-t from-[#f7f7f8] via-[#f7f7f8]/80 to-transparent dark:from-[#343541] dark:via-[#343541]/80"
                style={{zIndex: 10}}
            >
                <input
                    type="text"
                    className="flex-1 rounded-xl border border-[#ececf1] dark:border-[#444654] px-4 py-3 text-base bg-white dark:bg-[#40414f] text-gray-900 dark:text-[#ececf1] placeholder-gray-400 dark:placeholder-[#8e8ea0] focus:outline-none focus:ring-2 focus:ring-[#007aff] dark:focus:ring-[#007aff] transition shadow"
                    placeholder="메시지를 입력하세요... (Shift+Enter로 줄바꿈)"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            handleSend(e as unknown as React.FormEvent<HTMLFormElement>)
                        }
                    }}
                    disabled={mutation.isPending}
                />
                <button
                    type="submit"
                    className="bg-[#007aff] hover:bg-[#005bb5] text-white font-semibold px-6 py-3 rounded-xl transition shadow disabled:opacity-50 border border-[#007aff]"
                    disabled={!input.trim() || mutation.isPending}
                >
                    {mutation.isPending ? '전송 중...' : (
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2}
                             stroke="currentColor" className="w-5 h-5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7"/>
                        </svg>
                    )}
                </button>
            </form>
        </div>
    )
}
