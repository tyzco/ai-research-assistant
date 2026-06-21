<template>
  <div class="login-page">
    <!-- Left: Gradient Hero -->
    <div class="hero-panel">
      <div class="hero-bg" /><div class="hero-blur-1" /><div class="hero-blur-2" />
      <div class="hero-content">
        <div class="hero-logo"><span class="logo-icon">📚</span><span class="logo-text">Academic AI Assistant</span></div>
        <div class="hero-center">
          <h1 class="hero-title"><span class="shimmer-text">学术文献精读助手</span></h1>
          <p class="hero-subtitle">多源搜索 · RAG 深度问答 · Agent 自主调研</p>
          <div class="hero-stats">
            <div class="stat"><span class="stat-num">200+</span><span class="stat-label">单次论文发现</span></div>
            <div class="stat"><span class="stat-num">82%</span><span class="stat-label">Faithfulness</span></div>
            <div class="stat"><span class="stat-num">8</span><span class="stat-label">Agent 工具</span></div>
          </div>
        </div>
        <div class="hero-footer"><span>FastAPI + LanceDB + bge-small-zh</span></div>
      </div>
    </div>
    <!-- Right: Login Form -->
    <div class="login-panel">
      <div class="login-card">
        <div class="mobile-logo"><span class="logo-icon">📚</span><span>学术精读助手</span></div>
        <div class="login-header">
          <h2>{{ isRegister ? '创建账号' : '欢迎回来' }}</h2>
          <p>{{ isRegister ? '注册以使用完整功能' : '请输入账号密码登录' }}</p>
        </div>
        <el-form @submit.prevent="handleSubmit" class="login-form">
          <el-form-item><el-input v-model="username" placeholder="用户名" :prefix-icon="User" size="large" class="login-input"/></el-form-item>
          <el-form-item>
            <el-input v-model="password" :type="showPassword ? 'text' : 'password'" placeholder="密码" :prefix-icon="Lock" size="large" class="login-input">
              <template #suffix><el-icon @click="showPassword = !showPassword" style="cursor:pointer"><component :is="showPassword ? Hide : View"/></el-icon></template>
            </el-input>
          </el-form-item>
          <el-alert v-if="errorMsg" :title="errorMsg" type="error" show-icon :closable="false" style="margin-bottom:12px"/>
          <button type="submit" class="submit-btn" :disabled="loading">
            <span v-if="loading" class="loading-dots"><span class="dot"/><span class="dot"/><span class="dot"/></span>
            <span v-else>{{ isRegister ? '注册' : '登录' }}</span>
          </button>
        </el-form>
        <div class="login-footer">
          <el-button link @click="isRegister = !isRegister">{{ isRegister ? '已有账号？去登录' : '没有账号？去注册' }}</el-button>
          <el-divider/>
          <el-button link @click="skipLogin">跳过登录，直接使用</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/api'
import { ElMessage } from 'element-plus'
import { User, Lock, View, Hide } from '@element-plus/icons-vue'

const router = useRouter()
const username = ref(''); const password = ref('')
const isRegister = ref(false); const loading = ref(false)
const showPassword = ref(false); const errorMsg = ref('')

async function handleSubmit() {
  if (!username.value || !password.value) { errorMsg.value = '请输入用户名和密码'; return }
  errorMsg.value = ''; loading.value = true
  try {
    const endpoint = isRegister.value ? '/register' : '/login'
    const { data } = await api.post(endpoint, { username: username.value, password: password.value })
    localStorage.setItem('aia_token', data.access_token)
    localStorage.setItem('aia_user', data.user)
    api.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`
    ElMessage.success(isRegister.value ? '注册成功！' : '登录成功！')
    router.push('/dashboard')
  } catch (e: any) { errorMsg.value = e.response?.data?.detail || '操作失败' }
  loading.value = false
}
function skipLogin() { router.push('/dashboard') }
</script>

<style scoped>
.login-page{display:grid;grid-template-columns:1fr;min-height:100vh;overflow:hidden}
@media(min-width:1024px){.login-page{grid-template-columns:1fr 1fr}}
.hero-panel{display:none;position:relative;background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#a855f7 100%);color:#fff;overflow:hidden}
@media(min-width:1024px){.hero-panel{display:flex}}
.hero-bg{position:absolute;inset:0;background-image:radial-gradient(circle at 1px 1px,rgba(255,255,255,.05) 1px,transparent 0);background-size:24px 24px}
.hero-blur-1{position:absolute;top:25%;right:25%;width:256px;height:256px;background:rgba(255,255,255,.1);border-radius:50%;filter:blur(64px)}
.hero-blur-2{position:absolute;bottom:25%;left:25%;width:384px;height:384px;background:rgba(255,255,255,.05);border-radius:50%;filter:blur(64px)}
.hero-content{position:relative;z-index:10;display:flex;flex-direction:column;justify-content:space-between;padding:48px;width:100%}
.hero-logo{display:flex;align-items:center;gap:8px;font-size:18px;font-weight:600}
.logo-icon{font-size:32px}.logo-text{opacity:.9}
.hero-center{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}
.hero-title{font-size:28px;font-weight:700;margin-bottom:8px}
.shimmer-text{display:inline-block;background:linear-gradient(90deg,#fff 25%,#e2e8f0 50%,#fff 75%);background-size:200% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 2s linear infinite}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.hero-subtitle{font-size:14px;opacity:.8;margin-bottom:32px}
.hero-stats{display:flex;gap:32px}.stat{display:flex;flex-direction:column;align-items:center}
.stat-num{font-size:28px;font-weight:700}.stat-label{font-size:11px;opacity:.7}
.hero-footer{font-size:12px;opacity:.6;text-align:center}
.login-panel{display:flex;align-items:center;justify-content:center;padding:32px;background:var(--el-bg-color,#fff)}
.login-card{width:100%;max-width:420px}
.mobile-logo{display:flex;align-items:center;justify-content:center;gap:8px;font-size:18px;font-weight:600;margin-bottom:48px}
@media(min-width:1024px){.mobile-logo{display:none}}
.login-header{text-align:center;margin-bottom:32px}
.login-header h2{font-size:28px;font-weight:700;margin:0 0 4px}
.login-header p{color:#64748b;font-size:13px;margin:0}
.login-form{display:flex;flex-direction:column}
.submit-btn{width:100%;height:48px;border:none;border-radius:10px;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center}
.submit-btn:hover{box-shadow:0 8px 20px rgba(124,58,237,.3);transform:translateY(-1px)}
.submit-btn:disabled{opacity:.7;cursor:not-allowed}
.loading-dots{display:flex;gap:6px}.dot{width:8px;height:8px;background:#fff;border-radius:50%;animation:bounce 1.4s infinite ease-in-out both}
.dot:nth-child(1){animation-delay:0s}.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(.3);opacity:.3}40%{transform:scale(1);opacity:1}}
.login-footer{text-align:center;margin-top:16px}
</style>
