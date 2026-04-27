import questions from '../config/sample_questions.json'

export function useSampleQuestion() {
  const pick = (difficulty = 'hard') => {
    const pool = questions[difficulty] || questions.hard || []
    if (pool.length === 0) return ''
    const idx = Math.floor(Math.random() * pool.length)
    return pool[idx]
  }
  return { pick }
}
