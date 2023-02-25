import os, cv2, numpy, subprocess, tqdm, glob, sys
import pickle as pkl 
from scipy import signal
from scipy.io import wavfile
from TalkNet.talkNet import talkNet

class TalkNetWrapper():
	def __init__(self, videoPath, cacheDir):
		self.videoPath = videoPath
		self.cacheDir = cacheDir
		self.cropScale = 0.25
		self.audioFilePath = os.path.join(self.cacheDir, 'audio.wav')
		self.nDataLoaderThread = 10
		self.pretrainModel = '../TalkNet/pretrain_TalkSet.model'

	def readFaceTracks(self):
		faceTracksFile = os.path.join(self.cacheDir, 'face_retinaFace.pkl')
		faceTracks = pkl.load(open(faceTracksFile, 'rb'))
		allTracks = []
		for faceTrackId, faceTrack in faceTracks.items():
			frameNums = [int(round(face[0]*self.framesObj['fps'])) for face in faceTrack]
			boxes = []
			for box in faceTrack:
				x1 = int(round(box[1]*self.framesObj['width']))
				y1 = int(round(box[2]*self.framesObj['height']))
				x2 = int(round(box[3]*self.framesObj['width']))
				y2 = int(round(box[4]*self.framesObj['height']))
				boxes.append([x1, y1, x2, y2])
			allTracks.append({'frame':frameNums, 'bbox':boxes})
		return allTracks

	def crop_video(self, track, cropFile):
		# CPU: crop the face clips
		allFrames = self.framesObj['frames']
		vOut = cv2.VideoWriter(
			f'{cropFile}t.avi', cv2.VideoWriter_fourcc(*'XVID'), self.framesObj['fps'], (224, 224)
		)
		dets = {'x':[], 'y':[], 's':[]}
		for det in track['bbox']: # Read the tracks
			dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
			dets['y'].append((det[1]+det[3])/2) # crop center x 
			dets['x'].append((det[0]+det[2])/2) # crop center y
		dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
		dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
		dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
		for fidx, frame in enumerate(track['frame']):
			cs  = self.cropScale
			bs  = dets['s'][fidx]   # Detection box size
			bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
			image = allFrames[frame]
			frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
			my  = dets['y'][fidx] + bsi  # BBox center Y
			mx  = dets['x'][fidx] + bsi  # BBox center X
			face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
			vOut.write(cv2.resize(face, (224, 224)))
		audioTmp = f'{cropFile}.wav'
		audioStart  = (track['frame'][0]) / self.framesObj['fps']
		audioEnd    = (track['frame'][-1]+1) / self.framesObj['fps']
		vOut.release()
		#extract the audio file
		command = ("ffmpeg -y -nostdin -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
						(self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
		output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
		_, audio = wavfile.read(audioTmp)
		command = ("ffmpeg -y -nostdin -i %st.avi -i %s -threads %d -c:v copy -c:a copy %stt.avi -loglevel panic" % \
						(cropFile, audioTmp, self.nDataLoaderThread, cropFile)) # Combine audio and video file
		output = subprocess.call(command, shell=True, stdout=None)
		os.remove(f'{cropFile}t.avi')
		# convert to 25fps
		command = f'ffmpeg -y -nostdin -loglevel panic -i {cropFile}tt.avi -filter:v fps=25 {cropFile}.avi'
		output = subprocess.call(command, shell=True, stdout=None)
		os.remove(f'{cropFile}tt.avi')
		#extract the audio file from 25fps
		command = f'ffmpeg -y -nostdin -loglevel error -i {cropFile}.avi \
                -ar 16k -ac 1 {cropFile}.wav'
		output = subprocess.call(command, shell=True, stdout=None)
		return {'track':track, 'proc_track':dets}

	def evaluate_network(self, files, args):
		# GPU: active speaker detection by pretrained TalkNet
		s = talkNet()
		s.loadParameters(self.pretrainModel)
		sys.stderr.write("Model %s loaded from previous state! \r\n"%self.pretrainModel)
		s.eval()
		allScores = []
		# durationSet = {1,2,4,6} # To make the result more reliable
		durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
		for file in tqdm.tqdm(files, total = len(files)):
			fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
			_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
			audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
			video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
			videoFeature = []
			while video.isOpened():
				ret, frames = video.read()
				if ret == True:
					face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
					face = cv2.resize(face, (224,224))
					face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
					videoFeature.append(face)
				else:
					break
			video.release()
			videoFeature = numpy.array(videoFeature)
			length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
			audioFeature = audioFeature[:int(round(length * 100)),:]
			videoFeature = videoFeature[:int(round(length * 25)),:,:]
			allScore = [] # Evaluation use TalkNet
			for duration in durationSet:
				batchSize = int(math.ceil(length / duration))
				scores = []
				with torch.no_grad():
					for i in range(batchSize):
						inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
						inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
						embedA = s.model.forward_audio_frontend(inputA)
						embedV = s.model.forward_visual_frontend(inputV)	
						embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
						out = s.model.forward_audio_visual_backend(embedA, embedV)
						score = s.lossAV.forward(out, labels = None)
						scores.extend(score)
				allScore.append(scores)
			allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
			allScores.append(allScore)	
		return allScores

	def run(self):
		frameFilePath = os.path.join(self.cacheDir, 'frames.pkl')
		self.framesObj = pkl.load(open(frameFilePath, 'rb'))
		faceCropDir = os.path.join(self.cacheDir, 'face_crop_videos')
		os.makedirs(faceCropDir, exist_ok=True)
		allTracks = self.readFaceTracks()
		vidTracks = [
			self.crop_video(
				track,
				os.path.join(faceCropDir, '%05d' % ii),
			)
			for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks))
		]

		files = glob.glob(f'{faceCropDir}/*.avi')
		files.sort()

