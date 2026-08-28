import { writable } from 'svelte/store'

const networkErrorPattern = /failed to fetch|networkerror|load failed/i

export class JobController {
  constructor({ api, getError, setError }) {
    this.api = api
    this.getError = getError
    this.setError = setError
    this.sequence = 0
    this.current = {
      jobs: [],
      engineStates: { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline', media: 'offline', trainer: 'offline', upscale: 'offline', garment: 'offline', faceswap: 'offline' },
      refreshFailureCount: 0,
      refreshError: '',
      deletingJob: '',
      cancellingJob: '',
      retryingJob: ''
    }
    this.state = writable(this.current)
    this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  async refresh() {
    const sequence = ++this.sequence
    try {
      const [nextJobs, nextEngines] = await Promise.all([this.api.jobs(), this.api.engines()])
      if (sequence !== this.sequence) return false
      const jobs = [...nextJobs].sort((a, b) => {
        const createdDifference = Date.parse(b.created_at || 0) - Date.parse(a.created_at || 0)
        return createdDifference || String(b.id).localeCompare(String(a.id))
      })
      const error = this.getError()
      if (error === this.current.refreshError || networkErrorPattern.test(error)) this.setError('')
      this.setState({
        jobs,
        engineStates: Object.fromEntries(nextEngines.map((item) => [item.kind, item.status])),
        refreshFailureCount: 0,
        refreshError: ''
      })
      return true
    } catch {
      if (sequence !== this.sequence) return false
      const refreshFailureCount = this.current.refreshFailureCount + 1
      const refreshError = refreshFailureCount >= 3 ? 'Media API 연결이 끊겼습니다. 자동 재연결 중입니다.' : this.current.refreshError
      const previousRefreshError = this.current.refreshError
      this.setState({ refreshFailureCount, refreshError })
      const error = this.getError()
      if (refreshError && (!error || error === previousRefreshError || networkErrorPattern.test(error))) this.setError(refreshError)
      return false
    }
  }

  async deleteJob(job, confirmDelete) {
    if (!confirmDelete(`이 ${job.status === 'failed' ? '실패한 작업' : '작업'}과 저장 파일을 삭제할까요?`)) return false
    this.setState({ deletingJob: job.id })
    this.setError('')
    try {
      await this.api.deleteJob(job.id)
      await this.refresh()
      return true
    } catch (cause) {
      this.setError(cause.message)
      return false
    } finally {
      this.setState({ deletingJob: '' })
    }
  }

  async cancelJob(job) {
    this.setState({ cancellingJob: job.id })
    this.setError('')
    try {
      await this.api.cancelJob(job.id)
      await this.refresh()
      return true
    } catch (cause) {
      this.setError(cause.message)
      return false
    } finally {
      this.setState({ cancellingJob: '' })
    }
  }

  async retryJob(job) {
    this.setState({ retryingJob: job.id })
    this.setError('')
    try {
      await this.api.retryJob(job.id)
      await this.refresh()
      return true
    } catch (cause) {
      this.setError(cause.message)
      return false
    } finally {
      this.setState({ retryingJob: '' })
    }
  }

  async clearFinishedJobs(confirmDelete) {
    const count = this.current.jobs.filter((job) => job.status !== 'queued' && job.status !== 'running').length
    if (!count || !confirmDelete(`완료·실패·취소 작업 ${count}개와 저장 파일을 모두 삭제할까요?`)) return false
    this.setState({ deletingJob: 'all' })
    this.setError('')
    try {
      await this.api.deleteFinishedJobs()
      await this.refresh()
      return true
    } catch (cause) {
      this.setError(cause.message)
      return false
    } finally {
      this.setState({ deletingJob: '' })
    }
  }
}
