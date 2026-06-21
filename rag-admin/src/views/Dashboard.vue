<template>
  <div>
    <h2 style="margin-bottom:16px">📊 系统仪表盘</h2>
    <el-row :gutter="16">
      <el-col :span="6">
        <el-card shadow="hover"><div style="text-align:center">
          <div style="font-size:36px;color:#3b82f6">{{ appStore.topics.length }}</div>
          <div style="color:#64748b;margin-top:4px">课题总数</div>
        </div></el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover"><div style="text-align:center">
          <div style="font-size:36px;color:#22c55e">{{ totalPapers }}</div>
          <div style="color:#64748b;margin-top:4px">论文总数</div>
        </div></el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover"><div style="text-align:center">
          <div style="font-size:36px;color:#f59e0b">{{ readyTopics }}</div>
          <div style="color:#64748b;margin-top:4px">就绪课题</div>
        </div></el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover"><div style="text-align:center">
          <div style="font-size:36px;color:#8b5cf6">{{ appStore.apiStatus === 'online' ? '🟢' : '🔴' }}</div>
          <div style="color:#64748b;margin-top:4px">API 状态</div>
        </div></el-card>
      </el-col>
    </el-row>

    <h3 style="margin:20px 0 12px">🔗 快速调研入口</h3>
    <el-card style="margin-bottom:16px">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <span style="font-weight:600;font-size:14px">📚 学术文献精读助手</span>
          <span style="color:#64748b;font-size:12px;margin-left:8px">轻量版前端 (端口 8001)</span>
          <span :style="{color: apiOnline ? '#22c55e' : '#ef4444', fontSize: '11px', marginLeft: '8px'}">{{ apiOnline ? '🟢 已连接' : '🔴 未连接' }}</span>
        </div>
        <el-space>
          <el-button type="primary" @click="openLightweight">🔗 在新窗口打开</el-button>
          <el-button @click="checkApiConnection">🔄 检查连接</el-button>
        </el-space>
      </div>
    </el-card>

    <h3 style="margin:20px 0 12px">📋 评估指标</h3>
    <el-row :gutter="16">
      <el-col :span="8">
        <el-card shadow="hover"><div style="text-align:center">
          <el-progress type="dashboard" :percentage="82" color="#3b82f6" />
          <div style="font-size:13px;color:#64748b">Faithfulness</div>
        </div></el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover"><div style="text-align:center">
          <el-progress type="dashboard" :percentage="80" color="#22c55e" />
          <div style="font-size:13px;color:#64748b">Context Precision</div>
        </div></el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover"><div style="text-align:center">
          <el-progress type="dashboard" :percentage="82" color="#8b5cf6" />
          <div style="font-size:13px;color:#64748b">Semantic Recall@5</div>
        </div></el-card>
      </el-col>
    </el-row>

    <h3 style="margin:20px 0 12px">🔧 快速操作</h3>
    <el-card>
      <el-space>
        <el-button type="primary" @click="dialogVisible = true">+ 新建课题</el-button>
        <el-button @click="appStore.loadTopics()">🔄 刷新</el-button>
        <el-button @click="appStore.checkHealth()">🔍 检查 API</el-button>
      </el-space>
    </el-card>

    <el-dialog v-model="dialogVisible" title="新建课题" width="500px">
      <el-input v-model="newTopicQuery" placeholder="输入研究方向..." @keyup.enter="handleCreate" />
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleCreate" :loading="creating">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { createTopic } from '@/api'
import { ElMessage } from 'element-plus'

const appStore = useAppStore()
const apiOnline = ref(false)

async function checkApiConnection() {
  try {
    const res = await fetch('http://localhost:8001/health')
    apiOnline.value = res.ok
  } catch { apiOnline.value = false }
}

function openLightweight() {
  window.open('http://localhost:8001/', '_blank')
}

onMounted(() => { checkApiConnection() })
const dialogVisible = ref(false)
const newTopicQuery = ref('')
const creating = ref(false)

const totalPapers = computed(() => appStore.topics.reduce((s: number, t: any) => s + (t.papers || 0), 0))
const readyTopics = computed(() => appStore.topics.filter((t: any) => t.status === 'ready').length)

async function handleCreate() {
  if (!newTopicQuery.value) return
  creating.value = true
  try {
    await createTopic(newTopicQuery.value)
    ElMessage.success('课题创建成功')
    dialogVisible.value = false
    newTopicQuery.value = ''
    await appStore.loadTopics()
  } catch (e: any) {
    ElMessage.error('创建失败: ' + e.message)
  }
  creating.value = false
}
</script>
