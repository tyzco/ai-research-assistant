import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/dashboard' },
    { path: '/login', name: 'Login', component: () => import('@/views/Login.vue') },
    { path: '/dashboard', name: 'Dashboard', component: () => import('@/views/Dashboard.vue') },
    { path: '/topics', name: 'Topics', component: () => import('@/views/Topics.vue') },
    { path: '/topics/:id', name: 'TopicDetail', component: () => import('@/views/TopicDetail.vue') },
    { path: '/knowledge', name: 'Knowledge', component: () => import('@/views/Knowledge.vue') },
    { path: '/history', name: 'History', component: () => import('@/views/History.vue') },
    { path: '/settings', name: 'Settings', component: () => import('@/views/Settings.vue') },
  ]
})

export default router
