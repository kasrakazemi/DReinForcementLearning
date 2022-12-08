
test_size = 40000
train_size =test_size*2
gap=20
cursor=0
pred=[]
while cursor <len(dataX):
    if cursor+train_size+gap >= len(dataX):
        X_train=dataX[cursor:]
        Y_train=dataY[cursor:]
        print(f'Train index {cursor} to {len(dataX)}')
        pred.append(model.predict(X_train))
        model.evaluate(X_train,Y_train)
        break
    s_idx = cursor
    e_idx=cursor+train_size
    X_train=dataX[range(s_idx,e_idx)]
    Y_train=dataY[range(s_idx,e_idx)]
    print(f'Train index: {s_idx} to {e_idx}')
    s_idx = cursor+train_size+gap
    e_idx= min(cursor+train_size+gap+test_size,len(dataX))
    X_test=dataX[range(s_idx,e_idx)]
    Y_test=dataY[range(s_idx,e_idx)]
    print(f'Test index: {s_idx} to {e_idx}')
    cursor+=test_size
    model.fit(X_train,
          Y_train,
          validation_data=(X_test,
          Y_test),
          epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[model_checkpoint_callback])

    pred.append(model.predict(X_train))
    pred.append(model.predict(X_test))

    