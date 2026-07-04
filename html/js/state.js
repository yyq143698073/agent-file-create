/** Reactive state store — publish/subscribe pattern.
 *
 *  Replaces the global `S` object in app.js with a structured store
 *  that notifies subscribers on changes. This enables gradual migration:
 *  app.js continues to use `S`; new modules import `store`.
 *
 *  Usage:
 *    import { store } from './state.js';
 *    store.set('taskId', 'abc123');
 *    store.subscribe('taskId', (newVal, oldVal) => console.log(newVal));
 */

/** Create a reactive store with pub/sub notifications. */
function createStore(initial = {}) {
  const _listeners = {};      // key → Set<callback>
  let _state = { ...initial }; // shallow copy

  return {
    /** Get current value for a key, or entire state if no key. */
    get(key) {
      return key === undefined ? { ..._state } : _state[key];
    },

    /** Set one or more keys. Notifies subscribers for each changed key. */
    set(updates) {
      const old = { ..._state };
      for (const [k, v] of Object.entries(updates)) {
        _state[k] = v;
        if (v !== old[k] && _listeners[k]) {
          for (const cb of _listeners[k]) {
            try { cb(v, old[k]); } catch (e) { console.error('store listener error:', e); }
          }
        }
      }
    },

    /** Subscribe to changes for a key. Returns unsubscribe function. */
    on(key, cb) {
      if (!_listeners[key]) _listeners[key] = new Set();
      _listeners[key].add(cb);
      return () => _listeners[key].delete(cb);
    },

    /** Subscribe to ANY key change. Returns unsubscribe function. */
    onAny(cb) {
      return this.on('*', cb);
    },
  };
}

/** Singleton store instance — replaces global `S` in app.js.
 *
 *  Keys: taskId, pollTimer, history, kb, lastAbText, previewTab,
 *        kbTab, generating, _outlineDone, _sending, _evalData,
 *        _evalTab, _waitingFeedback, _feedbackStage, _satisfactionKey,
 *        _waitingRedo, _redoType, _redoBaseVersion, _versions,
 *        _selectedVersion, _compareData, _compareSourceTab, _showDiff,
 *        _cachedDiff, _lastSatisfactionStage, _lastSatisfactionPreview,
 *        _lastSatisfactionVersion, _editingSection, _contentStreaming,
 *        _tplEditor, _historyDirty, _satisfactionSubmitting,
 *        _pendingTemplate
 */
export const store = createStore({
  taskId: '',
  pollTimer: null,
  history: [],
  kb: 'default',
  lastAbText: '',
  previewTab: 'outline',
  kbTab: 'kb-query',
  generating: false,
  // Internal keys (prefixed with _) — not reactive by convention
  _outlineDone: false,
  _sending: false,
  _evalData: null,
  _evalTab: 'combined',
  _waitingFeedback: false,
  _feedbackStage: '',
  _satisfactionKey: '',
  _waitingRedo: false,
  _redoType: '',
  _redoBaseVersion: 0,
  _versions: { outline: [], content: [] },
  _selectedVersion: { outline: null, content: null },
  _compareData: null,
  _compareSourceTab: 'content',
  _showDiff: true,
  _cachedDiff: null,
  _lastSatisfactionStage: '',
  _lastSatisfactionPreview: '',
  _lastSatisfactionVersion: 0,
  _editingSection: null,
  _contentStreaming: false,
  _tplEditor: { currentName: '', content: '', variables: [], dirty: false },
  _historyDirty: false,
  _satisfactionSubmitting: false,
  _pendingTemplate: null,
});

export default store;
