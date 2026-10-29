'use strict';
const scenarios = {
  highway: {number:'01',title:'Interactive highway driving',label:'Highway driving',on:'media1',description:'Observe the vehicle’s motion and surrounding traffic in the highway environment.'},
  merge: {number:'02',title:'Joining the traffic stream',label:'Ramp merge',on:'media2',description:'A ramp-merging scenario demonstrates the vehicle’s interaction with traffic in the target lane.'},
  blocking: {number:'03',title:'Responding to an adversarial blocker',label:'Adversarial blocking',on:'media5',off:'media6',description:'With intent inference, the vehicle recognizes a competitive driver and yields to merge behind. Switch inference off to compare the original recording.',offDescription:'Without intent inference, the vehicle attempts to overtake the blocker and merge ahead, leading to a more confrontational interaction.'},
  cutin: {number:'04',title:'Recognizing a brief yielding opportunity',label:'Dense two-lane cut-in',on:'media4',off:'media7',description:'With intent inference, the vehicle identifies the yielding interaction and commits to the available gap. Switch inference off to inspect the hesitant behavior.',offDescription:'Without intent inference, repeated probing and retreating miss the yielding window; the original recording ends in a collision.'},
  instruction: {number:'05',title:'Adapting when passenger priorities change',label:'Changing preferences',on:'media8',description:'Passenger preferences change from comfort-oriented to urgent travel. The reasoning module translates the new instruction into updated decision parameters.'}
};
const tabs = [...document.querySelectorAll('[data-scene]')];
const video = document.querySelector('#simulation-video');
const modeButtons = [...document.querySelectorAll('[data-mode]')];
let selected = 'highway';
let mode = 'on';
function renderScenario() {
  const scene = scenarios[selected];
  const file = mode === 'off' && scene.off ? scene.off : scene.on;
  video.pause();
  video.poster = `assets/posters/${file}.jpg`;
  video.querySelector('source').src = `assets/videos/${file}.mp4`;
  video.setAttribute('aria-label', `Full ${scene.label} recording${scene.off ? `, intent inference ${mode}` : ''}`);
  video.load();
  document.querySelector('#scenario-label').textContent = scene.label;
  document.querySelector('#scene-number').textContent = `${scene.number} / 05`;
  document.querySelector('#scene-title').textContent = scene.title;
  document.querySelector('#scene-description').textContent = mode === 'off' && scene.off ? scene.offDescription : scene.description;
  document.querySelector('#inference-toggle').hidden = !scene.off;
  document.querySelector('#simulation-download').href = `assets/videos/${file}.mp4`;
  // Update the native fallback link along with the visible download link.
  video.querySelector('a').href = `assets/videos/${file}.mp4`;
  document.querySelector('#scenario-panel').setAttribute('aria-labelledby', `tab-${selected}`);
  tabs.forEach(tab => {
    const active = tab.dataset.scene === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  modeButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => { selected = tab.dataset.scene; mode = 'on'; renderScenario(); });
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); tabs[next].click(); tabs[next].focus(); }
  });
});
modeButtons.forEach(button => button.addEventListener('click', () => { mode = button.dataset.mode; renderScenario(); }));
// Keep playback focused on the selected experiment; never autoplay large recordings.
const allVideos = [...document.querySelectorAll('video')];
allVideos.forEach(current => current.addEventListener('play', () => allVideos.forEach(other => { if (other !== current) other.pause(); })));
