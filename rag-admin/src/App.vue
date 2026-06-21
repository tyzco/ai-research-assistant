<template>
  <el-container style="min-height:100vh">
    <el-aside width="220px" style="background:#1e293b;color:#fff">
      <div style="padding:16px;font-size:16px;font-weight:bold;border-bottom:1px solid #334155">
        📚 学术精读助手
      </div>
      <el-menu :default-active="$route.path" router background-color="#1e293b" text-color="#94a3b8" active-text-color="#60a5fa" style="border:none">
        <el-menu-item index="/dashboard"><el-icon><DataAnalysis /></el-icon>仪表盘</el-menu-item>
        <el-menu-item index="/topics"><el-icon><Collection /></el-icon>课题管理</el-menu-item>
        <el-menu-item index="/knowledge"><el-icon><Document /></el-icon>知识库</el-menu-item>
        <el-menu-item index="/history"><el-icon><ChatLineSquare /></el-icon>问答记录</el-menu-item>
        <el-menu-item index="/settings"><el-icon><Setting /></el-icon>系统配置</el-menu-item>
      </el-menu>
      <div style="position:absolute;bottom:12px;left:12px;font-size:11px;color:#64748b">
        API: {{ appStore.apiStatus === 'online' ? '🟢 在线' : '🔴 离线' }}
      </div>
    </el-aside>
    <el-container>
      <el-header style="background:#fff;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;padding:0 20px">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item v-for="(item, i) in breadcrumbs" :key="i">{{ item }}</el-breadcrumb-item>
        </el-breadcrumb>
      </el-header>
      <el-main style="background:#f8fafc">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { onMounted } from 'vue'

const route = useRoute()
const appStore = useAppStore()

onMounted(() => {
  appStore.checkHealth()
  appStore.loadTopics()
})

const breadcrumbs = computed(() => {
  const m: Record<string, string> = { dashboard: '仪表盘', topics: '课题管理', knowledge: '知识库', history: '问答记录', settings: '系统配置' }
  const name = String(route.name || '').toLowerCase()
  return [m[name] || name]
})
</script>
