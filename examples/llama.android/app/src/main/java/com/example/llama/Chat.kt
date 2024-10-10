package com.example.llama

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.llama.R
import com.example.llama.ChatMessage
import com.example.llama.role
import kotlinx.coroutines.launch
import java.time.Duration
import java.time.format.DateTimeFormatter

@Composable
fun ChatTopBar() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 16.dp)
    ) {
        Text(
            text = "Model Chat",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.weight(10f)
        )
    }

}

@Composable
//@Preview
fun ChatBottomBar(selectImage: () -> Unit, sendMessage: (String) -> Unit, generating: Boolean = false) {
    var text_message by remember {
        mutableStateOf("")
    }
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp),
        color = Color(color = 0xe9e0ef),
        shape = RoundedCornerShape(24.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier =  Modifier.padding(horizontal = 8.dp)
            ) {
                Spacer(modifier = Modifier.size(8.dp))
                OutlinedTextField(
                    value = text_message,
                    onValueChange = {
                        text_message = it
                    },
                    singleLine = true,
                    modifier = Modifier
                        .padding(horizontal = 6.dp)
                        .width(240.dp)
                        .height(52.dp)
                    ,
                    shape = RoundedCornerShape(24.dp)

                )
                IconButton(onClick = selectImage) {
                    Image(
                        painter = painterResource(id = R.drawable.gallery_bold),
                        contentDescription = "Select Image",
                    )
                }
                IconButton(
                    onClick = {
                        sendMessage(text_message)
                        text_message = ""
                    },
                ) {
                    Image(painter = painterResource(id = R.drawable.send_linear), contentDescription = "Send Text")

                }

            }

        }

    }



}

@Composable
fun ChatMessageBubble(message: ChatMessage) {
    Surface(
        modifier = Modifier.padding(8.dp),
        color = if (message.role == role.USER) Color(color = 0xffEEFFDE) else Color(color = 0xffFFE8E8),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(8.dp)
        ) {
            when (message.type) {
                "text" -> {
                    Text(text = message.content)
                }

                "image" -> {
                    AsyncImage(
                        model = ImageRequest.Builder(LocalContext.current)
                            .data(message.imageUri)
                            .crossfade(true)
                            .build(),
                        contentDescription = "Image",
                        placeholder = painterResource(id = R.drawable.gallery_bold),
                        modifier = Modifier.widthIn(max = 240.dp).heightIn(max = 240.dp),
//                        modifier = Modifier
//                            .size(size) // Set the size of the avatar
//                            .clip(CircleShape), // Clip the image to a circle
//                        contentScale = ContentScale. // Crop the image if it's not a square
                    )
                }
            }

        }
    }
}

@Composable
fun AvatarFromResource(modifier: Modifier = Modifier, resourceId: Int = R.drawable.profile_circle, size: Dp = 36.dp) {
    Image(
        painter = painterResource(id = resourceId),
        contentDescription = "Avatar",
        modifier = modifier
            .size(size) // Set the size of the avatar
            .clip(CircleShape), // Clip the image to a circle
        contentScale = ContentScale.Crop // Crop the image if it's not a square
    )
}

@Composable
fun ChatMessage(message: ChatMessage) {
    Box(
        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        if (message.role == role.USER) {
            Row(
                modifier = Modifier.padding(start = 48.dp)
            ) {
                ChatMessageBubble(message = message)
                AvatarFromResource(resourceId = R.drawable.profile_circle, size = 24.dp)
            }
        } else {
            Row(
                modifier = Modifier.padding(end = 48.dp)
            ) {
                AvatarFromResource(resourceId = R.drawable.bot_svgrepo_com, size = 24.dp)
                ChatMessageBubble(message = message)
            }
        }
    }
}

@Composable
fun Chat( messages: List<ChatMessage>, selectImage: () -> Unit,generating: Boolean = false, sendMessage: (String) -> Unit ) {
    val listState = rememberLazyListState()
//    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            ChatTopBar()
        },
        bottomBar = {
            ChatBottomBar(selectImage = selectImage, sendMessage = sendMessage, generating = generating)
        }
    ) {
            paddingValues: PaddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            //lazyColumn
            LazyColumn(state = listState) {
                itemsIndexed(messages) { index, message ->
                    val isSelf = message.role == role.USER
                    if(isSelf) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                            ChatMessage(
                                message = message,
                            )
                        }
                    }
                    else {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                            ChatMessage(
                                message = message,
                            )
                        }
                    }
                }
            }
        }
    }

}
