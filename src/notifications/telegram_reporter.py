import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

class MultiTargetTelegramReporter:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """Initialize Telegram reporter with bot token and chat ID"""
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("Telegram bot token and chat ID must be provided")
            
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.max_message_length = 4000  # Telegram limit is 4096
        
    def send_message(self, text: str, parse_mode: str = 'Markdown', report_type: str = 'alert') -> bool:
        """Send a message using simple HTTP request and log to database"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            success = response.json().get('ok', False)
            error_msg = None if success else f"API returned ok=False"
            
            # Log to database
            self._log_report_to_db(report_type, text, success, error_msg)
            
            return success
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Telegram HTTP error: {e}")
            self._log_report_to_db(report_type, text, False, str(e))
            return False
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            self._log_report_to_db(report_type, text, False, str(e))
            return False
    
    def _log_report_to_db(self, report_type: str, content: str, success: bool, error_message: str = None):
        """Log telegram report to database"""
        try:
            from src.database.operations import db_ops
            query = """
            INSERT INTO telegram_reports (report_type, content, success, error_message)
            VALUES (:report_type, :content, :success, :error_message)
            """
            with db_ops.db.get_session() as session:
                from sqlalchemy import text
                session.execute(text(query), {
                    'report_type': report_type,
                    'content': content[:5000],
                    'success': success,
                    'error_message': error_message
                })
                session.commit()
        except Exception as e:
            logging.error(f"Failed to log telegram report to database: {e}")
    
    def send_hourly_report(self, predictions: Dict[str, Dict], trigger_type: str = "scheduled") -> bool:
        """Send comprehensive hourly report with all predictions"""
        try:
            message = self._format_hourly_report(predictions, trigger_type)
            
            # Split message if too long
            if len(message) > self.max_message_length:
                messages = self._split_message(message)
                for msg in messages:
                    if not self.send_message(msg, report_type='hourly'):
                        return False
            else:
                return self.send_message(message, report_type='hourly')
                
            return True
            
        except Exception as e:
            logging.error(f"Error sending hourly report: {e}")
            return False
            
    def _format_hourly_report(self, predictions: Dict[str, Dict], trigger_type: str = "scheduled") -> str:
        """Format comprehensive hourly report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        # Add trigger context
        if trigger_type == "startup":
            message = f"🟢 *SYSTEM STARTUP - Initial Predictions*\n"
            message += f"🚀 BSE Predict is now running\n"
        elif trigger_type == "manual":
            message = f"🔧 *MANUAL PREDICTION RUN*\n"
            message += f"👤 Triggered by user request\n"
        elif trigger_type == "scheduled":
            message = f"⏰ *SCHEDULED HOURLY PREDICTION*\n"
            message += f"🔄 Automatic prediction at :05\n"
        else:
            message = f"🔮 *PREDICTION REPORT*\n"
            
        message += f"📅 {timestamp}\n"
        message += "─" * 30 + "\n\n"
        
        # Summary statistics
        total_predictions = 0
        successful_predictions = 0
        high_confidence_count = 0
        
        # Process each asset
        for symbol, data in predictions.items():
            if 'error' in data:
                message += f"❌ *{symbol}*: {data['error']}\n\n"
                continue
                
            message += f"📊 *{symbol}*\n"
            
            if 'current_price' in data:
                message += f"💰 Current: ${data['current_price']:,.2f}\n"
                
            if 'predictions' in data:
                for target, pred in data['predictions'].items():
                    total_predictions += 1
                    
                    if 'error' in pred:
                        message += f"   {target}: ❌ {pred['error']}\n"
                    else:
                        successful_predictions += 1
                        direction = pred.get('prediction', 'UNKNOWN')
                        confidence = pred.get('confidence', 0)
                        
                        # Direction emoji
                        dir_emoji = "🟢" if direction == "UP" else "🔴"
                        
                        # Confidence level
                        if confidence > 0.75:
                            high_confidence_count += 1
                            conf_emoji = "🔥"
                        elif confidence > 0.65:
                            conf_emoji = "✨"
                        else:
                            conf_emoji = "💫"
                            
                        message += f"   {target}: {dir_emoji} {direction} "
                        message += f"({confidence:.1%} {conf_emoji})\n"
                        
                        # Add probabilities if available
                        if 'up_probability' in pred and 'down_probability' in pred:
                            message += f"      ↗️ {pred['up_probability']:.1%} vs ↘️ {pred['down_probability']:.1%}\n"
                            
            message += "\n"
            
        # Summary footer
        message += "📈 *Summary*\n"
        message += f"✅ Successful: {successful_predictions}/{total_predictions}\n"
        
        if high_confidence_count > 0:
            message += f"🔥 High confidence: {high_confidence_count} predictions\n"
            
        # Model status
        if successful_predictions < total_predictions * 0.5:
            message += "\n⚠️ *Note*: Some models need more training data for better accuracy.\n"
        
        # Add footer with schedule info
        message += "\n" + "─" * 30 + "\n"
        message += "📅 *Schedule*:\n"
        message += "• Predictions: Every hour at :05\n"
        message += "• Model retraining: Daily at 06:00 UTC\n"
        message += "• Performance summary: Daily at 18:00 UTC"
            
        return message
        
    def send_high_confidence_alert(self, symbol: str, target: str, prediction: Dict) -> bool:
        """Send alert for high-confidence predictions"""
        try:
            confidence = prediction.get('confidence', 0)
            if confidence < 0.75:
                return False
                
            direction = prediction.get('prediction', 'UNKNOWN')
            up_prob = prediction.get('up_probability', 0)
            down_prob = prediction.get('down_probability', 0)
            
            # Craft alert message
            emoji = "🚨" if confidence > 0.85 else "⚡"
            dir_emoji = "🟢📈" if direction == "UP" else "🔴📉"
            
            message = f"{emoji} *HIGH CONFIDENCE ALERT*\n"
            message += f"🎯 Confidence threshold exceeded!\n"
            message += "─" * 30 + "\n\n"
            message += f"Asset: *{symbol}*\n"
            message += f"Target: *{target}*\n"
            message += f"Direction: {dir_emoji} *{direction}*\n"
            message += f"Confidence: *{confidence:.1%}*\n"
            message += f"Probabilities: ↗️ {up_prob:.1%} vs ↘️ {down_prob:.1%}\n"
            message += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
            message += f"\n📌 _This is an automated alert based on prediction confidence_"
            
            return self.send_message(message, report_type='daily')
            
        except Exception as e:
            logging.error(f"Error sending high confidence alert: {e}")
            return False
            
    def send_daily_performance_summary(self, performance_data: Dict) -> bool:
        """Send daily performance summary"""
        try:
            message = "📊 *DAILY PERFORMANCE SUMMARY*\n"
            message += f"🌅 End of day report\n"
            message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
            message += "─" * 30 + "\n\n"
            
            for symbol, targets in performance_data.items():
                message += f"*{symbol}*\n"
                
                for target, metrics in targets.items():
                    accuracy = metrics.get('accuracy', 0)
                    total = metrics.get('total_predictions', 0)
                    
                    # Performance emoji
                    if accuracy > 0.7:
                        perf_emoji = "🎯"
                    elif accuracy > 0.6:
                        perf_emoji = "✅"
                    else:
                        perf_emoji = "📊"
                        
                    message += f"  {target}: {accuracy:.1%} accuracy ({total} predictions) {perf_emoji}\n"
                    
                message += "\n"
                
            return self.send_message(message, report_type='daily')
            
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")
            return False
            
    def _split_message(self, message: str) -> List[str]:
        """Split long message into chunks"""
        if len(message) <= self.max_message_length:
            return [message]
            
        messages = []
        lines = message.split('\n')
        current_message = ""
        
        for line in lines:
            if len(current_message) + len(line) + 1 > self.max_message_length:
                messages.append(current_message.strip())
                current_message = line + '\n'
            else:
                current_message += line + '\n'
                
        if current_message:
            messages.append(current_message.strip())
            
        return messages