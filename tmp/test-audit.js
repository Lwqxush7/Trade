(() => {
  const issues = [];

  // 1. Modals test
  const modalIds = ['tradeDetailModal', 'imagePreviewModal', 'strategyModal', 'exportImportModal'];
  for (const id of modalIds) {
    const el = document.getElementById(id);
    if (!el) {
      issues.push('Modal bulunamadı: ' + id);
    }
  }

  // 2. Test Best / Worst Trade calculations with edge cases
  const originalTrades = [...(window.AppState?.trades || [])];
  try {
    // Case A: No trades
    window.AppState.trades = [];
    window.updateOverviewStats();
    const bestSub0 = document.getElementById('bestTradeSub')?.textContent || '';
    const worstSub0 = document.getElementById('worstTradeSub')?.textContent || '';
    if (!bestSub0.includes('Henüz kâr eden işlem yok')) {
      issues.push('Kâr eden işlem yokken bestTradeSub beklenenden farklı: ' + bestSub0);
    }
    if (!worstSub0.includes('Henüz zarar eden işlem yok')) {
      issues.push('Zarar eden işlem yokken worstTradeSub beklenenden farklı: ' + worstSub0);
    }

    // Case B: Only winning trades
    window.AppState.trades = [
      { id: 'w1', pair: 'EURUSD', direction: 'Long', result: 'Win', pnl: 250, status: 'Closed', datetime: '2026-11-01T10:00' },
      { id: 'w2', pair: 'GBPUSD', direction: 'Long', result: 'Win', pnl: 500, status: 'Closed', datetime: '2026-11-02T10:00' }
    ];
    window.updateOverviewStats();
    const worstSubWinsOnly = document.getElementById('worstTradeSub')?.textContent || '';
    if (!worstSubWinsOnly.includes('Henüz zarar eden işlem yok')) {
      issues.push('Yalnızca kârlı işlemler varken worstTradeSub: ' + worstSubWinsOnly);
    }

    // Case C: Only losing trades
    window.AppState.trades = [
      { id: 'l1', pair: 'EURUSD', direction: 'Long', result: 'Loss', pnl: -200, status: 'Closed', datetime: '2026-11-01T10:00' }
    ];
    window.updateOverviewStats();
    const bestSubLossesOnly = document.getElementById('bestTradeSub')?.textContent || '';
    if (!bestSubLossesOnly.includes('Henüz kâr eden işlem yok')) {
      issues.push('Yalnızca zararlı işlemler varken bestTradeSub: ' + bestSubLossesOnly);
    }
  } catch(e) {
    issues.push('Stats calculation error: ' + e.message);
  } finally {
    window.AppState.trades = originalTrades;
    window.updateOverviewStats();
  }

  // 3. Test Calendar month navigation
  try {
    const prevMonthBtn = document.getElementById('calPrevMonthBtn');
    const nextMonthBtn = document.getElementById('calNextMonthBtn');
    const todayBtn = document.getElementById('calTodayBtn');
    if (!prevMonthBtn || !nextMonthBtn || !todayBtn) {
      issues.push('Takvim navigasyon butonları eksik');
    } else {
      prevMonthBtn.click();
      nextMonthBtn.click();
      todayBtn.click();
    }
  } catch(e) {
    issues.push('Calendar nav error: ' + e.message);
  }

  // 4. Test Backtest Calendar month navigation
  try {
    const btPrevMonthBtn = document.getElementById('btCalPrevMonthBtn');
    const btNextMonthBtn = document.getElementById('btCalNextMonthBtn');
    const btTodayBtn = document.getElementById('btCalTodayBtn');
    if (!btPrevMonthBtn || !btNextMonthBtn || !btTodayBtn) {
      issues.push('Backtest Takvim navigasyon butonları eksik');
    } else {
      btPrevMonthBtn.click();
      btNextMonthBtn.click();
      btTodayBtn.click();
    }
  } catch(e) {
    issues.push('Backtest Calendar nav error: ' + e.message);
  }

  // 5. Check Trade Detail Modal with strategy
  try {
    const dummyTrade = {
      id: 'dt-1',
      pair: 'EURUSD',
      direction: 'Long',
      datetime: '2026-11-10T14:30',
      timeframe: 'M15',
      session: 'New York',
      strategy: 'Silver Bullet',
      entryPrice: 1.0850,
      exitPrice: 1.0900,
      stopLoss: 1.0820,
      risk: 1,
      pnl: 350,
      result: 'Win',
      notes: 'Test note',
      lessons: 'Test lesson',
      images: []
    };
    if (typeof window.showTradeDetail === 'function') {
      window.showTradeDetail(dummyTrade);
      const detailModal = document.getElementById('tradeDetailModal');
      if (!detailModal || !detailModal.classList.contains('active')) {
        issues.push('showTradeDetail çağrıldığında tradeDetailModal active olmadı');
      }
      // Check strategy info card
      const stratCard = document.getElementById('detailStrategyCard');
      const stratName = document.getElementById('detailStrategyName');
      if (!stratCard) {
        issues.push('detailStrategyCard elementi bulunamadı');
      }
      if (!stratName || !stratName.textContent.includes('Silver Bullet')) {
        issues.push('detailStrategyName doğru strateji adını içermiyor: ' + stratName?.textContent);
      }
      // Close modal
      if (typeof window.closeModal === 'function') {
        window.closeModal('tradeDetailModal');
      }
    }
  } catch(e) {
    issues.push('Trade detail modal test error: ' + e.message);
  }

  // 6. Check Trade Detail Modal without strategy
  try {
    const dummyTradeNoStrat = {
      id: 'dt-2',
      pair: 'GBPUSD',
      direction: 'Short',
      datetime: '2026-11-10T16:30',
      timeframe: 'H1',
      session: 'London',
      strategy: '',
      entryPrice: 1.2500,
      exitPrice: 1.2450,
      stopLoss: 1.2520,
      risk: 1,
      pnl: 200,
      result: 'Win',
      images: []
    };
    if (typeof window.showTradeDetail === 'function') {
      window.showTradeDetail(dummyTradeNoStrat);
      const stratName = document.getElementById('detailStrategyName');
      if (!stratName || !stratName.textContent.includes('Strateji belirtilmemiş')) {
        issues.push('Stratejisiz işlemde detailStrategyName "Strateji belirtilmemiş" içermiyor: ' + stratName?.textContent);
      }
      if (typeof window.closeModal === 'function') {
        window.closeModal('tradeDetailModal');
      }
    }
  } catch(e) {
    issues.push('Trade detail no strat test error: ' + e.message);
  }

  return issues;
})()
