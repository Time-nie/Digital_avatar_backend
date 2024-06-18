import os
import re
import uuid
import json
import random
import requests
import threading
import subprocess
from time import sleep
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from prometheus_flask_exporter import PrometheusMetrics
from random_username import generate_user_id
import logging

logger = logging.getLogger('family_education')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler('/home/nzq/Digital_avatar/logs/family_education.log', mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

from agent import education_agent
from accumulation import knowledge_accumulation
from text_to_voice import TextToSpeech
from voice_to_text import wenet_voice_to_text
from parent_profile import summarize_once_person_prompt, summarize_overall_personality

app = Flask(__name__, static_folder='parent_dist')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqldb://hyw:pcgvga.527@localhost/family_education?unix_socket=/var/run/mysqld/mysqld.sock&charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

metrics = PrometheusMetrics(app)

ongoing_chats_lock = threading.Lock()
ongoing_chats = {}

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != '' and not path.startswith('metrics') and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

print(app.url_map)

class Parent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=False, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    chats = db.relationship('Chat', backref='parent', lazy=True)
    info = db.Column(db.Text, default='')
    profile = db.Column(db.Text, default='')
    respond_strategy = db.Column(db.Text, default='')
    event_summary = db.Column(db.Text, default='')

class Expert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=False, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    chats = db.relationship('Chat', backref='expert', lazy=True)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), default='')
    parent_id = db.Column(db.Integer, db.ForeignKey('parent.id'), nullable=False)
    expert_id = db.Column(db.Integer, db.ForeignKey('expert.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_message_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True)
    status = db.Column(db.Integer, default=1)  # 0 - suspended, 1 - not checked, 2 - checked
    profile = db.Column(db.Text, default='')
    respond_strategy = db.Column(db.Text, default='')
    event_summary = db.Column(db.Text, default='')
    expert_score = db.Column(db.Float, default=0.0)
    expert_feedback = db.Column(db.Text, default='')
    parent_score = db.Column(db.Float, default=0.0)
    parent_feedback = db.Column(db.Text, default='')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    sender_type = db.Column(db.String(20), nullable=False)  # 'parent' or 'bot' or 'expert'
    sender_id = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    machine_score = db.Column(db.Float, default=0.0)
    expert_score = db.Column(db.Float, default=0.0)
    expert_feedback = db.Column(db.Text, default='')
    expert_revision = db.Column(db.Text, default='')

class LogicKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.Text, nullable=False)
    logics = db.relationship('Logic', backref='logic_key', lazy=True)

class Logic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emotional = db.Column(db.Text, nullable=False)
    focus = db.Column(db.Text, nullable=False)
    logic = db.Column(db.Text, nullable=False)
    logic_key_id = db.Column(db.Integer, db.ForeignKey('logic_key.id'), nullable=False)

class Verification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    code = db.Column(db.String(4), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def is_valid(self):
        # valid for 10 minutes
        return (datetime.utcnow() - self.timestamp).total_seconds() < 600

with app.app_context():
    db.create_all()

def generate_expert_reply(chat_id, parent_message_id):
    with app.app_context():
        parent_message = db.session.get(Message, parent_message_id)
        if parent_message:
            with ongoing_chats_lock:
                cur_content = ongoing_chats[chat_id]
            raw_reply = education_agent(cur_content, parent_message.sender_id, parent_message.chat_id)
            with ongoing_chats_lock:
                if not ongoing_chats[chat_id].endswith(parent_message.content):
                    logger.info('======有新消息来了，忽略当前回复======')
                    logger.debug(parent_message.content)
                    return
                logger.info('======当前没有新消息，进行回复======')
                del ongoing_chats[chat_id]
            score_match = re.search(r'<score>([\d.]+)', raw_reply)
            if score_match:
                machine_score = float(score_match.group(1))
                raw_reply = raw_reply[:score_match.start()]
            else:
                machine_score = 0.0
            logger.info('Machine Score: ' + str(machine_score))
            if machine_score < 0.5:
                chat = Chat.query.get(chat_id)
                chat.status = 0

            raw_reply_list = raw_reply.split('\n\n')
            for raw_reply_list_content in raw_reply_list:
                expert_message = Message(
                    chat_id=chat_id,
                    sender_type='bot',
                    sender_id=1,  # ToDo: set expert id
                    content=raw_reply_list_content,
                    timestamp=datetime.utcnow(),
                    machine_score=machine_score
                )
                db.session.add(expert_message)
                temp_messages = Message.query.filter_by(chat_id=chat_id, sender_type='system').all()
                for message in temp_messages:
                    if '等待分身/专家回复中' in message.content:
                        db.session.delete(message)
                db.session.commit()

            summarize_once_person_prompt(parent_message.sender_id, parent_message.chat_id)
            summarize_overall_personality(parent_message.sender_id)  # ToDo: update each time parent login in

def verify_code_helper(phone, code):
    verification = Verification.query.filter_by(phone=phone, code=code).first()
    return verification and verification.is_valid()

@app.route('/send_verification_code', methods=['POST'])
def send_verification_code():
    data = request.json
    phone = data.get('phone')
    if not phone:
        return jsonify({'success': False, 'message': 'Phone number is required'}), 400

    code = '{:04d}'.format(random.randint(0, 9999))
    verification = Verification.query.filter_by(phone=phone).first()
    if verification:
        verification.code = code
        verification.timestamp = datetime.utcnow()
    else:
        verification = Verification(phone=phone, code=code)
    db.session.add(verification)
    db.session.commit()

    verification_data = {
        'text': {
            'phones': [phone],
            'sign_id': 'qm_8776c0683f9c43f2a9441e3f85f01e84',
            'template_id': 'mb_6eb088986ff34f579c87fe990dee2103',
            'para': [code]
        },
        'type': 2308
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post('https://www.ijiaodui.com/scheduler/check/v1', headers=headers, data=json.dumps(verification_data))

    if response.status_code == 200:
        return jsonify({'success': True, 'message': 'Verification code sent'}), 200
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification code'}), response.status_code

@app.route('/create_parent', methods=['POST'])
def create_parent():
    data = request.json
    phone = data.get('phone')
    verification_code = data.get('verification_code')
    if not verify_code_helper(phone, verification_code):
        return jsonify({'success': False, 'message': 'Invalid verification code'}), 400

    existing_parent = Parent.query.filter_by(phone=phone).first()
    if existing_parent:
        return jsonify({'success': True, 'parent_id': existing_parent.id}), 200
    try:
        if data.get('username') =='':
            new_parent = Parent(
                username= generate_user_id() ,
                phone=phone,
                password_hash=data.get('password_hash')
            )
            db.session.add(new_parent)
            db.session.commit()
            return jsonify({'success': True, 'parent_id': new_parent.id,'message':'Registered successfully'}), 200
        else:
            new_parent = Parent(
                username=data.get('username') ,
                phone=phone,
                password_hash=data.get('password_hash')
            )  
            db.session.add(new_parent)
            db.session.commit()
            return jsonify({'success': True, 'parent_id': new_parent.id}), 200
    except IntegrityError:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Phone already exists'}), 409

@app.route('/create_expert', methods=['POST'])
def create_expert():
    data = request.json
    phone = data.get('phone')
    verification_code = data.get('verification_code')
    if not verify_code_helper(phone, verification_code):
        return jsonify({'success': False, 'message': 'Invalid verification code'}), 400

    existing_expert = Expert.query.filter_by(phone=phone).first()
    if existing_expert:
        return jsonify({'success': True, 'expert_id': existing_expert.id}), 200
    try:
        if data.get('username') =='':
            new_expert = Expert(
                username=generate_user_id(),
                phone=phone,
                password_hash=data.get('password_hash'),
            )
            db.session.add(new_expert)
            db.session.commit()
            return jsonify({'success': True, 'expert_id': new_expert.id,'message':'Registered successfully'}), 200
        else:
            new_expert = Expert(
                username=data.get('username'),
                phone=phone,
                password_hash=data.get('password_hash'),
            )
            db.session.add(new_expert)
            db.session.commit()
            return jsonify({'success': True, 'expert_id': new_expert.id}), 200
    except IntegrityError:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Phone already exists'}), 409

@app.route('/create_chat', methods=['POST'])
def create_chat():
    data = request.json

    new_chat = Chat(parent_id=data.get('parent_id'), expert_id=data.get('expert_id'))
    db.session.add(new_chat)
    db.session.commit()

    new_chat.title = 'Chat ' + str(new_chat.id)  # ToDo: generate title for this chat
    db.session.commit()

    chat_data = {
        'id': new_chat.id,
        'title': new_chat.title,
        'parent_id': new_chat.parent_id,
        'expert_id': new_chat.expert_id,
        'created_at': new_chat.created_at.isoformat() + 'Z',
        'last_message_timestamp': new_chat.last_message_timestamp.isoformat() + 'Z',
        'messages': [],
        'status': new_chat.status,
        'profile': new_chat.profile,
        'respond_strategy': new_chat.respond_strategy,
        'event_summary': new_chat.event_summary,
        'expert_score': new_chat.expert_score,
        'expert_feedback': new_chat.expert_feedback,
        'parent_score': new_chat.parent_score,
        'parent_feedback': new_chat.parent_feedback,
    }

    return jsonify({'success': True, 'chat': chat_data}), 200

@app.route('/create_message', methods=['POST'])
def create_message():
    data = request.json

    new_message = Message(
        chat_id=data.get('chat_id'),
        sender_type=data.get('sender_type'),  # 'parent' or 'expert'
        sender_id=data.get('sender_id'),
        content=data.get('content'),
        machine_score=10.0  # from person
    )
    db.session.add(new_message)
    db.session.flush()
  


    chat_id = new_message.chat_id
    chat = Chat.query.get(chat_id)
    if chat:
        chat.last_message_timestamp = new_message.timestamp
        db.session.commit()
    else:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Chat not found'}), 404

    if chat and chat.status != 0 and new_message.sender_type == 'parent':
        with ongoing_chats_lock:
            if chat_id in ongoing_chats:
                ongoing_chats[chat_id] += '\n\n\n' + new_message.content
            else:
                ongoing_chats[chat_id] = new_message.content
            temp_message = Message(
                chat_id=chat_id,
                sender_type='system',
                sender_id=0, 
                content='等待分身/专家回复中',
                machine_score=10.0  
            )
            db.session.add(temp_message)
            try:
                db.session.commit()
            except Exception as e:
                logger.error(f"Failed to commit temporary message: {str(e)}")
                db.session.rollback()
        threading.Thread(target=generate_expert_reply, args=(chat_id, new_message.id)).start()
    elif chat and new_message.sender_type == 'expert':
        threading.Thread(target=knowledge_accumulation, args=(chat_id,)).start()
        temp_messages = Message.query.filter_by(chat_id=chat_id, sender_type='system').all()
        for message in temp_messages:
            if '等待分身/专家回复中' in message.content:
                db.session.delete(message)
    logger.debug(new_message.content)

    return jsonify({'success': True, 'message_id': new_message.id}), 200

@app.route('/parents/<int:parent_id>', methods=['GET'])
def get_parent(parent_id):
    parent = Parent.query.get(parent_id)
    if parent:
        parent_data = {
            'id': parent.id,
            'username': parent.username,
            'phone': parent.phone,
            'info': parent.info,
            'profile': parent.profile,
            'respond_strategy': parent.respond_strategy,
            'event_summary': parent.event_summary
        }
        return jsonify({'success': True, 'parent': parent_data}), 200
    else:
        return jsonify({'success': False, 'message': 'Parent not found.'}), 404

@app.route('/parents/<int:parent_id>/get_chats', methods=['GET'])
def get_parent_chats(parent_id):
    chats = Chat.query.filter_by(parent_id=parent_id).all()
    chats_data = []
    if chats:
        for chat in chats:
            messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
            messages_data = [{
                'id': message.id,
                'chat_id': message.chat_id,
                'sender_type': message.sender_type,
                'sender_id': message.sender_id,
                'content': message.content,
                'timestamp': message.timestamp.isoformat() + 'Z',
                'machine_score': message.machine_score,
                'expert_score': message.expert_score,
                'expert_feedback': message.expert_feedback,
                'expert_revision': message.expert_revision,
            } for message in messages]
            chats_data.append({
                'id': chat.id,
                'title': chat.title,
                'parent_id': chat.parent_id,
                'expert_id': chat.expert_id,
                'created_at': chat.created_at.isoformat() + 'Z',
                'last_message_timestamp': chat.last_message_timestamp.isoformat() + 'Z',
                'messages': messages_data,
                'status': chat.status,
                'profile': chat.profile,
                'respond_strategy': chat.respond_strategy,
                'event_summary': chat.event_summary,
                'expert_score': chat.expert_score,
                'expert_feedback': chat.expert_feedback,
                'parent_score': chat.parent_score,
                'parent_feedback': chat.parent_feedback,
            })
    return jsonify({'success': True, 'chats': chats_data}), 200

@app.route('/parents/<int:parent_id>/set_info', methods=['POST'])
def set_parent_info(parent_id):
    data = request.json.get('parent_info')
    parent = Parent.query.get(parent_id)
    if parent:
        parent.info = data
        db.session.commit()
        return jsonify({'success': True, 'message': 'Parent info set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Parent not found.'}), 404


################ 获取家长列表 ##################
@app.route('/get_all_parent_ids', methods=['GET'])
def get_all_parent_ids():
    try:
        # 使用load_only来仅加载id字段，提高效率
        all_parent_ids = db.session.query(Parent.id).all()
        print(all_parent_ids)
        # 将查询结果转换为id列表
        parent_ids_list = [parent_id[0] for parent_id in all_parent_ids]
        print(parent_ids_list)
        return jsonify({"parent_ids": parent_ids_list}), 200
    except Exception as e:
        logger.error(f"Error occurred while fetching parent IDs: {str(e)}")
        return jsonify({"error": "An error occurred while fetching parent IDs"}), 500



@app.route('/parents/<int:parent_id>/get_info', methods=['GET'])
def get_parent_info(parent_id):
    parent = Parent.query.get(parent_id)
    if parent:
        return jsonify({'success': True, 'info': parent.info}), 200
    return jsonify({'success': False, 'message': 'Parent not found.'}), 404

@app.route('/parents/<int:parent_id>/set_modeling', methods=['POST'])
def set_parent_modeling(parent_id):
    data = request.json
    parent = Parent.query.get(parent_id)
    if parent:
        parent.profile = data.get('profile')
        parent.respond_strategy = data.get('respond_strategy')
        parent.event_summary = data.get('event_summary')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Set parent modeling successfully.'}), 200
    return jsonify({'success': False, 'message': 'Parent not found.'}), 404

@app.route('/parents/<int:parent_id>/get_modeling', methods=['GET'])
def get_parent_modeling(parent_id):
    parent = Parent.query.get(parent_id)
    if parent:
        return jsonify({'success': True, 'profile': parent.profile, 'respond_strategy': parent.respond_strategy, 'event_summary': parent.event_summary}), 200
    return jsonify({'success': False, 'message': 'Parent not found.'}), 404

@app.route('/experts/<int:expert_id>', methods=['GET'])
def get_expert(expert_id):
    expert = Expert.query.get(expert_id)
    if expert:
        expert_data = {
            'id': expert.id,
            'username': expert.username,
            'phone': expert.phone
        }
        return jsonify({'success': True, 'expert': expert_data}), 200
    else:
        return jsonify({'success': False, 'message': 'Expert not found.'}), 404

@app.route('/experts/<int:expert_id>/get_chats', methods=['GET'])
def get_expert_chats(expert_id):
    chats = Chat.query.filter_by(expert_id=expert_id).all()
    chats_data = []
    if chats:
        for chat in chats:
            messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
            messages_data = [{
                'id': message.id,
                'chat_id': message.chat_id,
                'sender_type': message.sender_type,
                'sender_id': message.sender_id,
                'content': message.content,
                'timestamp': message.timestamp.isoformat() + 'Z',
                'machine_score': message.machine_score,
                'expert_score': message.expert_score,
                'expert_feedback': message.expert_feedback,
                'expert_revision': message.expert_revision,
            } for message in messages]
            chats_data.append({
                'id': chat.id,
                'title': chat.title,
                'parent_id': chat.parent_id,
                'expert_id': chat.expert_id,
                'created_at': chat.created_at.isoformat() + 'Z',
                'last_message_timestamp': chat.last_message_timestamp.isoformat() + 'Z',
                'messages': messages_data,
                'status': chat.status,
                'profile': chat.profile,
                'respond_strategy': chat.respond_strategy,
                'event_summary': chat.event_summary,
                'expert_score': chat.expert_score,
                'expert_feedback': chat.expert_feedback,
                'parent_score': chat.parent_score,
                'parent_feedback': chat.parent_feedback,
            })
    return jsonify({'success': True, 'chats': chats_data}), 200

@app.route('/experts/<int:expert_id>/get_parents', methods=['GET'])
def get_experts_parents(expert_id):
    expert = Expert.query.get(expert_id)
    if not expert:
        return jsonify({'success': False, 'message': 'Expert not found.'}), 404

    chats = Chat.query.filter_by(expert_id=expert_id).all()
    parent_ids = {chat.parent_id for chat in chats}

    parents = Parent.query.filter(Parent.id.in_(parent_ids)).all()
    parents_data = [{
        'id': parent.id,
        'username': parent.username,
        'phone': parent.phone,
        'info': parent.info,
        'profile': parent.profile,
        'respond_strategy': parent.respond_strategy,
        'event_summary': parent.event_summary
    } for parent in parents]

    return jsonify({'success': True, 'parents': parents_data}), 200

@app.route('/chats/expert/<int:expert_id>/parent/<int:parent_id>', methods=['GET'])
def get_chats_between_expert_and_parent(expert_id, parent_id):
    chats = Chat.query.filter_by(expert_id=expert_id, parent_id=parent_id).all()
    chats_data = []
    for chat in chats:
        messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
        messages_data = [{
            'id': message.id,
            'chat_id': message.chat_id,
            'sender_type': message.sender_type,
            'sender_id': message.sender_id,
            'content': message.content,
            'timestamp': message.timestamp.isoformat() + 'Z',
            'machine_score': message.machine_score,
            'expert_score': message.expert_score,
            'expert_feedback': message.expert_feedback,
            'expert_revision': message.expert_revision,
        } for message in messages]

        chats_data.append({
            'id': chat.id,
            'title': chat.title,
            'parent_id': chat.parent_id,
            'expert_id': chat.expert_id,
            'created_at': chat.created_at.isoformat() + 'Z',
            'last_message_timestamp': chat.last_message_timestamp.isoformat() + 'Z',
            'messages': messages_data,
            'status': chat.status,
            'profile': chat.profile,
            'respond_strategy': chat.respond_strategy,
            'event_summary': chat.event_summary,
            'expert_score': chat.expert_score,
            'expert_feedback': chat.expert_feedback,
            'parent_score': chat.parent_score,
            'parent_feedback': chat.parent_feedback,
        })

    return jsonify({'success': True, 'chats': chats_data}), 200

@app.route('/chats/<int:chat_id>/get_messages', methods=['GET'])
def get_chat_messages(chat_id):
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.asc()).all()
    if messages:
        messages_data = [{
            'id': message.id,
            'chat_id': message.chat_id,
            'sender_type': message.sender_type,
            'sender_id': message.sender_id,
            'content': message.content,
            'timestamp': message.timestamp.isoformat() + 'Z',
            'machine_score': message.machine_score,
            'expert_score': message.expert_score,
            'expert_feedback': message.expert_feedback,
            'expert_revision': message.expert_revision,
        } for message in messages]
        return jsonify({'success': True, 'messages': messages_data}), 200
    else:
        return jsonify({'success': True, 'messages': []}), 200

@app.route('/chats/<int:chat_id>/set_suspended', methods=['POST'])
def set_chat_suspended(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        chat.status = 0
        db.session.commit()
        return jsonify({'success': True, 'message': 'Set chat suspended successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found or failed to set.'}), 404

@app.route('/chats/<int:chat_id>/set_not_checked', methods=['POST'])
def set_chat_not_checked(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        chat.status = 1
        db.session.commit()
        return jsonify({'success': True, 'message': 'Set chat not checked successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found or failed to set.'}), 404

@app.route('/chats/<int:chat_id>/set_checked', methods=['POST'])
def set_chat_checked(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        chat.status = 2
        db.session.commit()
        return jsonify({'success': True, 'message': 'Set chat checked successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found or failed to set.'}), 404

@app.route('/chats/<int:chat_id>/set_modeling', methods=['POST'])
def set_chat_modeling(chat_id):
    data = request.json
    chat = Chat.query.get(chat_id)
    if chat:
        chat.profile = data.get('profile')
        chat.respond_strategy = data.get('respond_strategy')
        chat.event_summary = data.get('event_summary')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Set chat modeling successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/get_modeling', methods=['GET'])
def get_chat_modeling(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        return jsonify({'success': True, 'profile': chat.profile, 'respond_strategy': chat.respond_strategy, 'event_summary': chat.event_summary}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/set_expert_score_and_feedback', methods=['POST'])
def set_chat_expert_score_and_feedback(chat_id):
    data = request.json
    chat = Chat.query.get(chat_id)
    if chat:
        chat.expert_score = data.get('expert_score')
        chat.expert_feedback = data.get('expert_feedback')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Expert score and feedback set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/get_expert_score', methods=['GET'])
def get_chat_expert_score(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        return jsonify({'success': True, 'expert_score': chat.expert_score}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/get_expert_feedback', methods=['GET'])
def get_chat_expert_feedback(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        return jsonify({'success': True, 'expert_feedback': chat.expert_feedback}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/set_parent_score_and_feedback', methods=['POST'])
def set_chat_parent_score_and_feedback(chat_id):
    data = request.json
    chat = Chat.query.get(chat_id)
    if chat:
        chat.parent_score = data.get('parent_score')
        chat.parent_feedback = data.get('parent_feedback')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Parent score and feedback set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/get_parent_score', methods=['GET'])
def get_chat_parent_score(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        return jsonify({'success': True, 'parent_score': chat.parent_score}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/chats/<int:chat_id>/get_parent_feedback', methods=['GET'])
def get_chat_parent_feedback(chat_id):
    chat = Chat.query.get(chat_id)
    if chat:
        return jsonify({'success': True, 'parent_feedback': chat.parent_feedback}), 200
    return jsonify({'success': False, 'message': 'Chat not found.'}), 404

@app.route('/messages/<int:message_id>/set_expert_score', methods=['POST'])
def set_message_expert_score(message_id):
    data = request.json
    message = Message.query.get(message_id)
    if message:
        message.expert_score = data.get('expert_score')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Expert score set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/messages/<int:message_id>/get_expert_score', methods=['GET'])
def get_message_expert_score(message_id):
    message = Message.query.get(message_id)
    if message:
        return jsonify({'success': True, 'expert_score': message.expert_score}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/messages/<int:message_id>/set_expert_feedback', methods=['POST'])
def set_message_expert_feedback(message_id):
    data = request.json
    message = Message.query.get(message_id)
    if message:
        message.expert_feedback = data.get('expert_feedback')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Expert feedback set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/messages/<int:message_id>/get_expert_feedback', methods=['GET'])
def get_message_expert_feedback(message_id):
    message = Message.query.get(message_id)
    if message:
        return jsonify({'success': True, 'expert_feedback': message.expert_feedback}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/messages/<int:message_id>/set_expert_revision', methods=['POST'])
def set_message_expert_revision(message_id):
    data = request.json
    message = Message.query.get(message_id)
    if message:
        message.expert_revision = data.get('expert_revision')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Expert revision set successfully.'}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/messages/<int:message_id>/get_expert_revision', methods=['GET'])
def get_message_expert_revision(message_id):
    message = Message.query.get(message_id)
    if message:
        return jsonify({'success': True, 'expert_revision': message.expert_revision}), 200
    return jsonify({'success': False, 'message': 'Message not found.'}), 404

@app.route('/logics/add', methods=['POST'])
def add_logic():
    data = request.json
    key = data.get('key')
    logic_key = LogicKey.query.filter_by(key=key).first()
    if not logic_key:
        logic_key = LogicKey(key=key)
        db.session.add(logic_key)
        db.session.commit()
    new_logic = Logic(
        emotional=data.get('emotional'),
        focus=data.get('focus'),
        logic=data.get('logic'),
        logic_key_id=logic_key.id
    )
    db.session.add(new_logic)
    db.session.commit()
    return jsonify({'success': True, 'logic_id': new_logic.id}), 200

@app.route('/logics/get_all', methods=['GET'])
def get_all_logics():
    logic_keys = LogicKey.query.all()
    logics = [{
        'id': logic_key.id,
        'key': logic_key.key,
        'logics': [{
            'id': logic.id,
            'emotional': logic.emotional,
            'focus': logic.focus,
            'logic': logic.logic
        } for logic in logic_key.logics]
    } for logic_key in logic_keys]
    return jsonify({'success': True, 'logic_keys': logics}), 200

### methods not related to database
@app.route('/convert_text_to_audio', methods=['POST'])
def convert_text_to_audio():
    text = request.json.get('text')
    unique_filename = str(uuid.uuid4())
    audio_path = f'/home/hyw/FamilyEducation/Digital_avatar/backend_mock/audio_tmp/{unique_filename}.wav'

    app = TextToSpeech(text)
    app.get_token()
    app.save_audio(audio_path)

    # ToDo: change to a method that does not block the thread
    sleep_time = 0
    while not os.path.exists(audio_path):
        sleep(1)
        sleep_time += 1
        if sleep_time > 20:
            return jsonify({'error': 'Timeout waiting for audio file to be saved'}), 504
    return send_file(
        audio_path,
        mimetype='audio/wav',
        as_attachment=True
    )

@app.route('/convert_audio_to_text', methods=['POST'])
def convert_audio_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    unique_filename = str(uuid.uuid4())
    webm_audio_path = f'/home/hyw/FamilyEducation/Digital_avatar/backend_mock/audio_tmp/{unique_filename}.webm'
    wav_audio_path = f'/home/hyw/FamilyEducation/Digital_avatar/backend_mock/audio_tmp/{unique_filename}.wav'

    audio_file = request.files['audio']
    audio_file.save(webm_audio_path)
    convert_to_wav(webm_audio_path, wav_audio_path)
    text = wenet_voice_to_text(wav_audio_path)

    os.remove(webm_audio_path)
    os.remove(wav_audio_path)

    return jsonify({'text': text})

def convert_to_wav(input_path, output_path):
    command = ['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path]
    subprocess.run(command, check=True)


@app.route('/parents/<int:parent_id>/update_username', methods=['POST'])
def update_username(parent_id):
    data = request.json
    new_username = data.get('username')
    if not new_username:
        return jsonify({'success': False, 'message': 'New username is required'}), 400

    parent = Parent.query.get(parent_id)
    if not parent:
        return jsonify({'success': False, 'message': 'Parent not found.'}), 404

    parent.username = new_username
    try:
        db.session.commit()
        return jsonify({'success': True, 'message': 'Username updated successfully.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Failed to update username due to an error.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=22500, debug=True)
    # with app.app_context():
    #     print("All parent IDs:", get_experts_parents(1))
