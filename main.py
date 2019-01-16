from Spectrogram import *
from Network import *


global_step = 1


# Initialize a counter for GenA and GenB losses, to ease printing when the network is training 
A = 100
B = 100
genA = 0
genB = 0

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    # Restore a session if needed
    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint("/home/kraev_simeon/summaries/summaries_long_train/low2/")
        saver.restore(session, chkpt_fname)

    for epoch in range(epochs):   
        print ("Epoch", epoch)

        # Given a path to a folder with files, create a list with all the paths and shuffle it.
        
        path_list_A = Path_List(path_A)
        shuffle(path_list_A)
        path_list_B = Path_List(path_B) 
        shuffle(path_list_B)
            
            
        # A way to keep track how far in a song the spectrogram function is 
        index_A  = 0
        offset_A = 0
        index_B = 0
        offset_B = 0 
        
        # While there are still song paths in the list make spectrograms 
        
        while len(path_list_A) >1 and len(path_list_B) > 1:
            
            # Make Spectrograms and remember how far you are in a song 
            
            spec_list_A, path_list_A, offset_A = SpecMaker(path_list_A, spec_length, index_A, offset_A)
            spec_list_B, path_list_B, offset_B = SpecMaker(path_list_B, spec_length, index_B, offset_B)

            print(global_step, end="\r", flush=True)
            
            # Optimizing the G_A network
            
            _, fake_B_temp, summary_str, genA = session.run([g_A_trainer, gen_B, g_A_loss_summ,g_loss_A],
                                                   feed_dict={input_A:spec_list_A,
                                                             input_B:spec_list_B })      
            
            

            train_writer.add_summary(summary_str,global_step = global_step)                
            
            
            fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)
            
            
            # Optimizing the D_B network
            
            _, summary_str = session.run([d_B_trainer, d_B_loss_summ],feed_dict={input_A:spec_list_A,
                                                                                 input_B:spec_list_B,
                                                                                 fake_pool_B:fake_B_temp1})
            

            train_writer.add_summary(summary_str,global_step = global_step)
                    
                    
            # Optimizing the G_B network
                                
            _, fake_A_temp, summary_str, genB = session.run([g_B_trainer, gen_A, g_B_loss_summ,g_loss_B],
                                                   feed_dict={input_A:spec_list_A,
                                                              input_B:spec_list_B })
                
            train_writer.add_summary(summary_str,global_step = global_step)
                    
                    
            fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)

            # Optimizing the D_A network
            
            _, summary_str = session.run([d_A_trainer, d_A_loss_summ],feed_dict={input_A:spec_list_A,
                                                                                 input_B:spec_list_B,
                                                                                 fake_pool_A:fake_A_temp1 })


            train_writer.add_summary(summary_str,global_step = global_step)
            
            num_fake_inputs += 1
            
            # If the error is at a new low print it out and save
            
            if genA < A and genB < B:
                A = genA
                B = genB
                print('Loss for A is',A,'Loss for B is',B)
                
                saver.save(session, '/home/kraev_simeon/summaries/summaries_long_train/low2/')
                            
            if global_step % 10 == 0:
                saver.save(session,'/home/kraev_simeon/summaries/summaries_long_train/')
            
            global_step += 1 
                                
            epoch += 1

        train_writer.add_graph(session.graph)