import { writable } from 'svelte/store'
import { framesForDuration, snapDimension } from './videoTiming.js'

const emptyUsage = {
  cpu_percent: null,
  gpu_percent: null,
  mem_percent: null,
  mem_used_gb: null,
  mem_total_gb: null
}

const initialState = {
  systemUsage: emptyUsage,
  videoModelStatus: null,
  imageCheckpointStatus: null,
  preparingVideoModels: false,
  preparingImageCheckpoints: false,
  convertingImageCheckpoints: false,
  savingDownloadCredentials: false,
  savingConfig: false,
  storage: null,
  cleaningStorage: false
}

export class SettingsController {
  constructor({ api, setError, setMessage, setBusy = () => {} }) {
    this.api = api
    this.setError = setError
    this.setMessage = setMessage
    this.setBusy = setBusy
    this.current = { ...initialState }
    this.state = writable(this.current)
    this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  async refreshSystemUsage() {
    try {
      this.setState({ systemUsage: await this.api.system() })
    } catch {
      this.setState({ systemUsage: { ...emptyUsage } })
    }
  }

  async refreshVideoModels() {
    try { this.setState({ videoModelStatus: await this.api.videoModels() }) }
    catch { this.setState({ videoModelStatus: null }) }
  }

  async refreshImageCheckpoints() {
    try { this.setState({ imageCheckpointStatus: await this.api.imageCheckpoints() }) }
    catch { this.setState({ imageCheckpointStatus: null }) }
  }

  async prepareImageCheckpoints({ civitaiToken, hfToken, variants }) {
    if (!variants.length) {
      this.setError('준비할 Krea 체크포인트를 하나 이상 선택하세요.')
      return { clearTokens: false }
    }
    this.setState({ preparingImageCheckpoints: true })
    this.setError(''); this.setMessage('')
    try {
      const status = await this.api.prepareImageCheckpoints(civitaiToken.trim(), hfToken.trim(), variants)
      this.setState({ imageCheckpointStatus: status })
      this.setMessage(status.started ? 'Krea 체크포인트 준비를 시작했습니다.' : '이미 모델 준비가 진행 중입니다.')
      await this.refreshImageCheckpoints()
      return { clearTokens: true }
    } catch (cause) {
      this.setError(cause.message)
      return { clearTokens: false }
    } finally {
      this.setState({ preparingImageCheckpoints: false })
    }
  }

  async convertImageCheckpoints({ civitaiToken, variants, removeBF16Sources }) {
    if (!variants.length) {
      this.setError('변환할 Krea 체크포인트를 하나 이상 선택하세요.')
      return { clearCivitaiToken: false }
    }
    this.setState({ convertingImageCheckpoints: true })
    this.setError(''); this.setMessage('')
    try {
      const status = await this.api.convertImageCheckpointsNVFP4(civitaiToken.trim(), variants, removeBF16Sources)
      this.setState({ imageCheckpointStatus: status })
      this.setMessage(status.started ? 'BF16 다운로드와 NVFP4 변환을 시작했습니다.' : '이미 변환 작업이 진행 중입니다.')
      await this.refreshImageCheckpoints()
      return { clearCivitaiToken: true }
    } catch (cause) {
      this.setError(cause.message)
      return { clearCivitaiToken: false }
    } finally {
      this.setState({ convertingImageCheckpoints: false })
    }
  }

  async prepareVideoModels(hfToken) {
    this.setState({ preparingVideoModels: true })
    this.setError(''); this.setMessage('')
    try {
      const status = await this.api.prepareVideoModels(hfToken.trim())
      this.setState({ videoModelStatus: status })
      this.setMessage(status.ready && status.a2v_ready
        ? 'LTX 일반 영상과 A2V 모델이 이미 준비되어 있습니다.'
        : '모델 준비를 시작했습니다. 이 화면에서 진행 상태를 확인할 수 있습니다.')
      await this.refreshVideoModels()
      return { clearHFToken: true }
    } catch (cause) {
      this.setError(cause.message)
      return { clearHFToken: false }
    } finally {
      this.setState({ preparingVideoModels: false })
    }
  }

  async saveDownloadCredentials(civitaiToken, hfToken) {
    if (!civitaiToken.trim() && !hfToken.trim()) return { clearTokens: false }
    this.setState({ savingDownloadCredentials: true })
    this.setError(''); this.setMessage('')
    try {
      await this.api.saveLoraTokens(civitaiToken.trim(), hfToken.trim())
      this.setMessage('다운로드 인증 정보를 저장했습니다.')
      await Promise.all([this.refreshImageCheckpoints(), this.refreshVideoModels()])
      return { clearTokens: true }
    } catch (cause) {
      this.setError(cause.message)
      return { clearTokens: false }
    } finally {
      this.setState({ savingDownloadCredentials: false })
    }
  }

  async loadStorage() {
    try {
      const storage = await this.api.storage()
      this.setState({ storage })
      return storage
    } catch (cause) {
      this.setError(cause.message)
      return null
    }
  }

  async cleanupStorage(confirmDelete, formatBytes) {
    const amount = formatBytes(this.current.storage?.reclaimable_bytes || 0)
    if (!confirmDelete(`실행 중인 작업을 제외한 임시 파일 ${amount}을(를) 삭제할까요?`)) return false
    this.setState({ cleaningStorage: true })
    this.setError(''); this.setMessage('')
    try {
      const result = await this.api.cleanupTemporaryStorage()
      const storage = await this.api.storage()
      this.setState({ storage })
      this.setMessage(`임시 폴더 ${result.removed_directories}개, ${formatBytes(result.removed_bytes)}을(를) 정리했습니다.`)
      return true
    } catch (cause) {
      this.setError(cause.message)
      return false
    } finally {
      this.setState({ cleaningStorage: false })
    }
  }

  async saveConfig(settings, durationSeconds) {
    const normalized = structuredClone(settings)
    normalized.image.default_width = snapDimension(normalized.image.default_width, 8, 256, 2048)
    normalized.image.default_height = snapDimension(normalized.image.default_height, 8, 256, 2048)
    normalized.video.default_width = snapDimension(normalized.video.default_width, 64, 256, 1920)
    normalized.video.default_height = snapDimension(normalized.video.default_height, 64, 256, 1920)
    normalized.video.default_frames = framesForDuration(durationSeconds, normalized.video.default_fps)
    this.setState({ savingConfig: true })
    this.setBusy(true)
    this.setError(''); this.setMessage('')
    try {
      const result = await this.api.saveConfig(normalized)
      this.setMessage(result.restart_required
        ? '저장했습니다. Listen 주소 또는 데이터 폴더 변경은 Media 재시작 후 적용됩니다.'
        : '저장했습니다. API 연결과 생성 기본값이 즉시 적용됐습니다.')
      return result
    } catch (cause) {
      this.setError(cause.message)
      return null
    } finally {
      this.setBusy(false)
      this.setState({ savingConfig: false })
    }
  }
}
