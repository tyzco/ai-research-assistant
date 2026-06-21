import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchTopics, healthCheck } from '@/api'

export const useAppStore = defineStore('app', () => {
  const topics = ref<any[]>([])
  const currentTopic = ref<any>(null)
  const loading = ref(false)
  const apiStatus = ref('unknown')

  async function loadTopics() {
    try {
      topics.value = await fetchTopics()
    } catch (e) { /* offline */ }
  }

  async function checkHealth() {
    try {
      await healthCheck()
      apiStatus.value = 'online'
    } catch {
      apiStatus.value = 'offline'
    }
  }

  return { topics, currentTopic, loading, apiStatus, loadTopics, checkHealth }
})
