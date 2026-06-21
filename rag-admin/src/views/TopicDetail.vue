<template>
  <div>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
      <el-button @click="$router.back()" :icon="'ArrowLeft'">返回</el-button>
      <h2>{{ topic?.query || '课题详情' }}</h2>
      <el-tag :type="topic?.status==='ready'?'success':'warning'">{{ topic?.status==='ready'?'就绪':'构建中' }}</el-tag>
    </div>

    <el-row :gutter="16">
      <el-col :span="8"><el-card><div style="text-align:center"><div style="font-size:28px;color:#3b82f6">{{ topic?.papers || 0 }}</div><div>论文数</div></div></el-card></el-col>
      <el-col :span="8"><el-card><div style="text-align:center"><div style="font-size:28px;color:#22c55e">{{ topic?.total_papers || 0 }}</div><div>总记录</div></div></el-card></el-col>
      <el-col :span="8"><el-card><div style="text-align:center"><div style="font-size:28px;color:#8b5cf6">{{ topic?.total_images || 0 }}</div><div>图片</div></div></el-card></el-col>
    </el-row>

    <div style="margin-top:16px;display:flex;gap:8px">
      <el-button type="primary" @click="exportMD">⬇ 导出 Markdown</el-button>
      <el-button @click="deleteTopic" type="danger" plain>删除课题</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { exportTopic } from '@/api'

const route = useRoute()
const appStore = useAppStore()
const topic = ref<any>(null)

onMounted(() => {
  const id = String(route.params.id)
  topic.value = appStore.topics.find((t: any) => t.topic_id === id)
})

async function exportMD() {
  if (!topic.value) return
  try {
    const blob = await exportTopic(topic.value.topic_id)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob as any)
    a.download = `topic_${topic.value.topic_id}.md`
    a.click()
  } catch {}
}

function deleteTopic() {
  topic.value = null
  appStore.topics = appStore.topics.filter((t: any) => t.topic_id !== topic.value?.topic_id)
}
</script>
