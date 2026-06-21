import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8001',
  timeout: 60000,
})

export async function fetchTopics() {
  const { data } = await api.get('/topics')
  return data
}

export async function createTopic(query: string) {
  const { data } = await api.post('/create_topic', { query })
  return data
}

export async function searchPapers(query: string, keywords_en?: string[], keywords_cn?: string[]) {
  const { data } = await api.post('/search_papers', { query, keywords_en, keywords_cn })
  return data
}

export async function importPapers(urls: string[], topicId: string) {
  const { data } = await api.post('/download_bulk', { urls, topic_id: topicId })
  return data
}

export async function askQuestion(topicId: string, question: string) {
  const { data } = await api.post('/ask', { topic_id: topicId, question })
  return data
}

export async function fetchTopicStatus(topicId: string) {
  const { data } = await api.get(`/topic_status/${topicId}`)
  return data
}

export async function exportTopic(topicId: string) {
  const { data } = await api.get(`/export/${topicId}`, { responseType: 'blob' })
  return data
}

export async function uploadPDF(topicId: string, files: File[]) {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  const { data } = await api.post(`/upload_pdf/${topicId}`, fd)
  return data
}

export async function runAgent(task: string, maxSteps = 3) {
  const { data } = await api.post('/agent/run', { task, max_steps: maxSteps })
  return data
}

export async function healthCheck() {
  const { data } = await api.get('/health')
  return data
}

export default api
