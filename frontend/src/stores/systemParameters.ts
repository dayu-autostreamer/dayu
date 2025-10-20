// filepath: /Users/onecheck/PycharmProjects/dayu-inner-dev/frontend/src/stores/systemParameters.ts
import { defineStore } from 'pinia'
import { useInstallStateStore } from '/@/stores/installState'

export type SystemParamItem = {
  id: string
  data: Record<string, any>
}
export type SystemTask = {
  timestamp: number | string
  data: SystemParamItem[]
}

const LOCAL_LOG_KEY = 'system_parameters_buffer_v1'

export const useSystemParametersStore = defineStore('system_parameters', {
  state: () => ({
    bufferedTaskCache: [] as SystemTask[],
    maxBufferedTaskCacheSize: 20 as number,
    pollingInterval: null as any,
    initialized: false as boolean,
  }),
  actions: {
    init() {
      if (this.initialized) return
      this.initialized = true
      // load cache from storage
      this.loadFromStorage()
      // start/stop by backend state on boot
      this.syncWithBackendInstallState()
      // subscribe to install store for runtime changes
      try {
        const installStore = useInstallStateStore()
        installStore.$subscribe((mutation, state) => {
          if (state.status === 'install') this.startPolling()
          else this.stopPolling()
        })
      } catch {}
    },

    async syncWithBackendInstallState() {
      try {
        const resp = await fetch('/api/install_state')
        const json = await resp.json()
        const state = json?.state
        if (state === 'install') this.startPolling(); else this.stopPolling()
      } catch {
        // ignore network errors
      }
    },

    loadFromStorage() {
      try {
        const raw = localStorage.getItem(LOCAL_LOG_KEY)
        if (!raw) return
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed)) {
          const slice = parsed.slice(-this.maxBufferedTaskCacheSize)
          this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length, ...slice)
        }
      } catch {}
    },

    persistToStorage() {
      try {
        localStorage.setItem(LOCAL_LOG_KEY, JSON.stringify(this.bufferedTaskCache))
      } catch {}
    },

    async fetchLatest() {
      try {
        const response = await fetch('/api/system_parameters')
        const data = await response.json()
        const newTasks: SystemTask[] = (data || []).map((task: any) => ({
          ...task,
          data: (task.data || []).map((item: any) => ({
            id: String(item.id),
            data: item.data
          }))
        }))
        const merged = [...this.bufferedTaskCache, ...newTasks]
        const sliced = merged.slice(-this.maxBufferedTaskCacheSize)
        // update in place to retain reactivity
        this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length, ...sliced)
        this.persistToStorage()
      } catch (e) {
        // swallow errors to keep polling
        // console.error('system param fetch failed', e)
      }
    },

    startPolling() {
      if (this.pollingInterval) return
      // immediate fetch then interval
      this.fetchLatest()
      this.pollingInterval = setInterval(() => this.fetchLatest(), 2000)
    },

    stopPolling() {
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval)
        this.pollingInterval = null
      }
    },
  }
})

