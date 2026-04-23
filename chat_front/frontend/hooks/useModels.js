import { useEffect, useState } from 'react'

const _host = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const _isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:'
const MODELS_URL = `${_isSecure ? 'https' : 'http'}://${_host}:10001/models`

const FALLBACK_MODELS = ['Gemma4-E4B-it', 'GaussO4', 'GaussO4-think']

export function useModels() {
  const [models, setModels] = useState(FALLBACK_MODELS)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(MODELS_URL)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.models) && data.models.length > 0) {
          setModels(data.models)
        }
      })
      .catch(() => { /* keep fallback */ })
      .finally(() => setLoading(false))
  }, [])

  return { models, loading }
}
