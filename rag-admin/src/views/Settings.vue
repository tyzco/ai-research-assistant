<template>
  <div>
    <h2 style="margin-bottom:16px">⚙️ 系统配置</h2>
    <el-card style="margin-bottom:16px">
      <template #header>LLM 模型</template>
      <el-select v-model="llmModel" style="width:300px">
        <el-option label="DeepSeek V3 (deepseek-chat)" value="deepseek-chat" />
        <el-option label="DeepSeek R1 (deepseek-reasoner)" value="deepseek-reasoner" />
      </el-select>
      <el-button style="margin-left:8px" @click="save('llm')">保存</el-button>
    </el-card>
    <el-card style="margin-bottom:16px">
      <template #header>视觉模型</template>
      <el-select v-model="visionModel" style="width:300px">
        <el-option label="Qwen-VL-Max" value="qwen-vl-max" />
        <el-option label="Qwen-VL-Plus" value="qwen-vl-plus" />
        <el-option label="GLM-4V" value="glm-4v" />
        <el-option label="关闭" value="" />
      </el-select>
      <el-button style="margin-left:8px" @click="save('vision')">保存</el-button>
    </el-card>
    <el-card style="margin-bottom:16px">
      <template #header>嵌入模型</template>
      <el-input v-model="embedModel" placeholder="BAAI/bge-small-zh" style="width:300px" />
      <div style="font-size:12px;color:#64748b;margin-top:4px">需重建知识库后生效</div>
    </el-card>
    <el-card>
      <template #header>多模态图片筛选</template>
      <el-switch v-model="multimodalEnabled" active-text="开启" inactive-text="关闭" />
      <div style="margin-top:8px">
        <span style="font-size:12px;color:#64748b">最小图片尺寸 (px)</span>
        <el-input-number v-model="minImgSize" :min="50" :max="500" style="margin-left:8px" />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
const llmModel = ref(localStorage.getItem('aia_m') || 'deepseek-chat')
const visionModel = ref(localStorage.getItem('aia_v') || 'qwen-vl-max')
const embedModel = ref('BAAI/bge-small-zh')
const multimodalEnabled = ref(true)
const minImgSize = ref(100)
function save(type: string) {
  if (type === 'llm') localStorage.setItem('aia_m', llmModel.value)
  if (type === 'vision') localStorage.setItem('aia_v', visionModel.value)
  ElMessage.success('已保存')
}
</script>
