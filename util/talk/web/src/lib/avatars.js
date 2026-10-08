export const avatarPresets = [
  { id: 'spark', name: '스파크', url: '/avatars/spark.png' },
  { id: 'orbit', name: '오브', url: '/avatars/orbit.png' },
  { id: 'earth', name: '지구', url: '/avatars/earth.png' },
  { id: 'saturn', name: '토성', url: '/avatars/saturn.png' },
  { id: 'robot', name: '로봇', url: '/avatars/robot.png' },
  { id: 'quantum-computer', name: '양자컴퓨터', url: '/avatars/quantum-computer.png' },
  { id: 'person-blue', name: '블루 인물', url: '/avatars/person-blue.png' },
  { id: 'person-warm', name: '웜 인물', url: '/avatars/person-warm.png' },
  { id: 'cat', name: '고양이', url: '/avatars/cat.png' },
  { id: 'dog', name: '강아지', url: '/avatars/dog.png' },
  { id: 'bear', name: '곰', url: '/avatars/bear.png' },
  { id: 'rabbit', name: '토끼', url: '/avatars/rabbit.png' },
];

export function avatarURL(value, fallback = 'spark') {
  if (value?.startsWith('/api/images/')) return value;
  const id = value === 'preset:computer'
    ? 'quantum-computer'
    : value?.startsWith('preset:') ? value.slice(7) : fallback;
  return avatarPresets.find((preset) => preset.id === id)?.url
    || avatarPresets.find((preset) => preset.id === fallback)?.url
    || avatarPresets[0].url;
}

export function avatarImageID(value) {
  return value?.startsWith('/api/images/') ? value.slice('/api/images/'.length) : '';
}
