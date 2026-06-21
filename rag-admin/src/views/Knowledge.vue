<template>
  <div>
    <h2 style="margin-bottom:16px">📄 知识库管理</h2>
    <el-row :gutter="16">
      <el-col :span="12" v-for="t in readyTopics" :key="t.topic_id">
        <el-card shadow="hover" style="margin-bottom:12px">
          <div style="font-weight:600">{{ t.query }}</div>
          <el-descriptions :column="2" size="small" style="margin:8px 0">
            <el-descriptions-item label="论文">{{ t.papers }} 篇</el-descriptions-item>
            <el-descriptions-item label="状态">✅ 就绪</el-descriptions-item>
          </el-descriptions>
          <el-progress :percentage="100" color="#22c55e" style="margin:8px 0" />
        </el-card>
      </el-col>
    </el-row>
    <el-empty v-if="!readyTopics.length" description="没有就绪的知识库，请先在课题中构建" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'
const appStore = useAppStore()
const readyTopics = computed(() => (appStore.topics || []).filter((t: any) => t.status === 'ready'))
</script>
