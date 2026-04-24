import questions from '../config/sample_questions.json'

export function useSampleQuestion() {
  const pick = () => {
    const idx = Math.floor(Math.random() * questions.length)
    return questions[idx]
  }
  return { pick }
}
