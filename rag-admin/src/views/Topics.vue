<template>
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
      <h2>📁 课题管理</h2>
      <el-button type="primary" @click="showCreate = true">+ 新建课题</el-button>
    </div>

    <el-row :gutter="16">
      <el-col :span="8" v-for="t in appStore.topics" :key="t.topic_id">
        <el-card shadow="hover" style="margin-bottom:12px;cursor:pointer" @click="$router.push('/topics/'+t.topic_id)">
          <div style="font-weight:600;font-size:14px;margin-bottom:8px">{{ t.query || '未命名课题' }}</div>
          <el-space>
            <el-tag :type="t.status==='ready'?'success':'warning'" size="small">{{ t.status==='ready'?'就绪':'构建中' }}</el-tag>
            <span style="font-size:12px;color:#64748b">{{ t.papers || 0 }} 篇论文</span>
          </el-space>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-if="!appStore.topics.length" description="暂无课题" />

    <el-dialog v-model="showCreate" title="新建课题" width="500px">
      <el-input v-model="query" placeholder="输入研究方向..." @keyup.enter="create" />
      <template #footer>
        <el-button @click="showCreate=false">取消</el-button>
        <el-button type="primary" @click="create" :loading="loading">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from '@/stores/app'
import { createTopic } from '@/api'
import { ElMessage } from 'element-plus'

const appStore = useAppStore()
const showCreate = ref(false)
const query = ref('')
const loading = ref(false)

async function create() {
  if (!query.value) return
  loading.value = true
  try {
    await createTopic(query.value)
    ElMessage.success('课题创建成功')
    showCreate.value = false; query.value = ''
    await appStore.loadTopics()
  } catch (e: any) {
    ElMessage.error('创建失败: ' + e.message)
  }
  loading.value = false
}
</script>
