<template>
  <div>
    <h2 style="margin-bottom:16px">💬 问答记录</h2>
    <el-select v-model="selectedTopic" placeholder="选择课题" style="width:300px;margin-bottom:16px" @change="loadHistory">
      <el-option v-for="t in appStore.topics" :key="t.topic_id" :label="t.query||'未命名'" :value="t.topic_id" />
    </el-select>
    <el-empty v-if="!selectedTopic" description="请先选择一个课题" />
    <div v-else>
      <el-alert title="提示" type="info" description="问答记录将按对话顺序展示" show-icon style="margin-bottom:12px" />
      <el-timeline>
        <el-timeline-item v-for="(msg, i) in history" :key="i" :type="msg.role==='user'?'primary':'success'"
          :timestamp="msg.role==='user'?'👤 用户':'🤖 AI'">
          <el-card><div style="white-space:pre-wrap;font-size:13px">{{ msg.content?.substring(0,500) }}</div></el-card>
        </el-timeline-item>
      </el-timeline>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from '@/stores/app'
const appStore = useAppStore()
const selectedTopic = ref('')
const history = ref<any[]>([])

function loadHistory() {
  history.value = [
    { role: 'user', content: '示例问题：请搜索人脸识别的相关论文' },
    { role: 'assistant', content: '（实际历史从后端 message_store 加载，此处为占位）' },
  ]
}
</script>
