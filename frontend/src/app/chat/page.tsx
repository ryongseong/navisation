"use client";

import { Suspense } from 'react'
import ChatPage from './ChatClient'

export default function Page() {
  return (
    <Suspense fallback={<div>Loading chat...</div>}>
      <ChatPage />
    </Suspense>
  )
}